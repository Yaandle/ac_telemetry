#!/usr/bin/env python3
"""
Telemetry ML Trainer - Baseline Models for Control Prediction
Train and evaluate simple models to predict driver inputs from state
UPDATED: Uses feature_schema.py for consistency and validation
"""

import sys
import numpy as np
from pathlib import Path
from typing import Tuple, Dict
import argparse
import joblib

try:
    from sklearn.linear_model import LinearRegression
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
except ImportError:
    print("❌ Missing scikit-learn. Install with: pip install scikit-learn")
    sys.exit(1)

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Import feature schema
try:
    from feature_schema import (
        POLICY_FEATURES, ACTION_LABELS,
        validate_state, validate_normalization
    )
except ImportError:
    print("❌ Missing feature_schema.py. Ensure it's in the same directory.")
    sys.exit(1)

# ============================================================================
# MODEL DEFINITIONS
# ============================================================================

class BaselineModel:
    """Base class for control prediction models"""
    
    def __init__(self, name: str):
        self.name = name
        self.model_gas = None
        self.model_brake = None
        self.model_steer = None
        self.metrics = {}
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train separate models for each action"""
        raise NotImplementedError
    
    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """Predict actions for test data"""
        gas_pred = self.model_gas.predict(X_test).reshape(-1, 1)
        brake_pred = self.model_brake.predict(X_test).reshape(-1, 1)
        steer_pred = self.model_steer.predict(X_test).reshape(-1, 1)
        steer_pred = np.clip(steer_pred, -1.0, 1.0)
        return np.hstack([gas_pred, brake_pred, steer_pred])
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """Evaluate model performance"""
        y_pred = self.predict(X_test)
        
        metrics = {}
        
        for i, action in enumerate(ACTION_LABELS):
            mse = mean_squared_error(y_test[:, i], y_pred[:, i])
            rmse = np.sqrt(mse)
            r2 = r2_score(y_test[:, i], y_pred[:, i])
            
            metrics[f'{action}_rmse'] = rmse
            metrics[f'{action}_r2'] = r2
        
        # Overall metrics
        metrics['overall_rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
        metrics['overall_r2'] = r2_score(y_test, y_pred)
        
        self.metrics = metrics
        return metrics

class LinearModel(BaselineModel):
    """Simple linear regression baseline"""
    
    def __init__(self):
        super().__init__("Linear Regression")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        print(f"  Training {self.name}...")
        self.model_gas = LinearRegression().fit(X_train, y_train[:, 0])
        self.model_brake = LinearRegression().fit(X_train, y_train[:, 1])
        self.model_steer = LinearRegression().fit(X_train, y_train[:, 2])

class MLPModel(BaselineModel):
    """Multi-layer perceptron (neural network) model"""
    
    def __init__(self, hidden_layers=(64, 32)):
        super().__init__(f"MLP {hidden_layers}")
        self.hidden_layers = hidden_layers
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        print(f"  Training {self.name}...")
        
        self.model_gas = MLPRegressor(
            hidden_layer_sizes=self.hidden_layers,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            verbose=False
        ).fit(X_train, y_train[:, 0])
        
        self.model_brake = MLPRegressor(
            hidden_layer_sizes=self.hidden_layers,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            verbose=False
        ).fit(X_train, y_train[:, 1])
        
        self.model_steer = MLPRegressor(
            hidden_layer_sizes=self.hidden_layers,
            max_iter=500,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
            verbose=False
        ).fit(X_train, y_train[:, 2])

# ============================================================================
# TRAINING PIPELINE
# ============================================================================

def load_dataset(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load preprocessed dataset with validation"""
    X = np.load(data_dir / "X_states_normalized.npy")
    y = np.load(data_dir / "y_actions.npy")
    
    # Load normalization arrays for validation
    norm_min = np.load(data_dir / "normalization_min.npy")
    norm_max = np.load(data_dir / "normalization_max.npy")
    
    # Ensure features are within normalization bounds
    X = np.clip(X, norm_min, norm_max)

    # CRITICAL VALIDATION
    print(f"\n🔍 Validating dataset...")
    print(f"  Features shape: {X.shape}")
    print(f"  Actions shape: {y.shape}")
    print(f"  Expected features: {len(POLICY_FEATURES)}")
    
    # Validate feature count
    try:
        validate_state(X[0], "training dataset")
        print(f"  ✅ Feature count validated: {X.shape[1]} features")
    except AssertionError as e:
        print(f"\n❌ FEATURE MISMATCH ERROR:\n{e}")
        print("\nPlease regenerate the dataset with the updated telemetry_analysis.py")
        sys.exit(1)
    
    # Validate normalization arrays
    try:
        validate_normalization(norm_min, norm_max)
        print(f"  ✅ Normalization arrays validated")
    except AssertionError as e:
        print(f"\n❌ NORMALIZATION ERROR:\n{e}")
        sys.exit(1)
    
    # Validate action count
    if y.shape[1] != len(ACTION_LABELS):
        print(f"\n❌ Action count mismatch!")
        print(f"  Expected: {len(ACTION_LABELS)} actions")
        print(f"  Got: {y.shape[1]} actions")
        sys.exit(1)
    
    print(f"  ✅ Action labels validated: {ACTION_LABELS}")
    print(f"  ✅ All validations passed\n")
    
    return X, y

def train_and_evaluate_models(data_dir: Path, test_size: float = 0.2):
    """Train and compare baseline models"""
    
    print("\n" + "="*70)
    print("🤖 ML MODEL TRAINING - Feature Schema Validated")
    print("="*70)
    
    # Load data with validation
    print("\n📂 Loading dataset...")
    X, y = load_dataset(data_dir)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    print(f"  Train samples: {len(X_train):,}")
    print(f"  Test samples: {len(X_test):,}")
    
    # Define models
    models = [
        LinearModel(),
        MLPModel(hidden_layers=(64, 32)),
        MLPModel(hidden_layers=(128, 64, 32)),
        MLPModel(hidden_layers=(256, 128, 64))
    ]
    
    # Train and evaluate
    print("\n🎯 Training models...")
    results = {}
    
    for model in models:
        model.train(X_train, y_train)
        metrics = model.evaluate(X_test, y_test)
        results[model.name] = metrics
        
        print(f"\n  ✅ {model.name}")
        print(f"     Gas RMSE:   {metrics['gas_rmse']:.4f}  (R²: {metrics['gas_r2']:.3f})")
        print(f"     Brake RMSE: {metrics['brake_rmse']:.4f}  (R²: {metrics['brake_r2']:.3f})")
        print(f"     Steer RMSE: {metrics['steer_rmse']:.4f}  (R²: {metrics['steer_r2']:.3f})")
        print(f"     Overall R²: {metrics['overall_r2']:.3f}")
    
    # Find best model
    best_model = max(models, key=lambda m: m.metrics['overall_r2'])
    
    print("\n" + "="*70)
    print(f"🏆 Best Model: {best_model.name}")
    print(f"   Overall R²: {best_model.metrics['overall_r2']:.3f}")
    print("="*70)
    
    return models, X_test, y_test

def plot_predictions(models, X_test: np.ndarray, y_test: np.ndarray, 
                    output_dir: Path, sample_size: int = 1000):
    """Plot predicted vs actual traces"""
    
    print("\n📊 Generating prediction visualizations...")
    
    output_dir.mkdir(exist_ok=True)
    
    # Sample data for plotting
    sample_indices = np.random.choice(len(X_test), min(sample_size, len(X_test)), replace=False)
    sample_indices = np.sort(sample_indices)
    
    X_sample = X_test[sample_indices]
    y_sample = y_test[sample_indices]
    
    action_colors = ['#2ecc71', '#e74c3c', '#9b59b6']
    
    for model in models:
        y_pred = model.predict(X_sample)
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 10))
        fig.suptitle(f'{model.name} - Predictions vs Actual', fontsize=16, fontweight='bold')
        
        for i, (action, color) in enumerate(zip(ACTION_LABELS, action_colors)):
            ax = axes[i]
            
            # Plot actual
            ax.plot(y_sample[:, i], label='Actual', color='black', 
                   linewidth=1.5, alpha=0.7)
            
            # Plot predicted
            ax.plot(y_pred[:, i], label='Predicted', color=color, 
                   linewidth=1.5, alpha=0.7)
            
            ax.set_ylabel(action.capitalize(), fontsize=12, fontweight='bold')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
            
            # Add RMSE to plot
            rmse = model.metrics[f'{action}_rmse']
            r2 = model.metrics[f'{action}_r2']
            ax.text(0.02, 0.95, f'RMSE: {rmse:.4f}\nR²: {r2:.3f}', 
                   transform=ax.transAxes, fontsize=10,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        axes[-1].set_xlabel('Sample Index', fontsize=12)
        plt.tight_layout()
        
        # Save
        filename = output_dir / f"{model.name.replace(' ', '_').lower()}_predictions.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ Saved: {filename}")

def plot_model_comparison(models, output_dir: Path):
    """Create comparison bar chart of model performance"""
    
    print("\n📈 Generating model comparison chart...")
    
    model_names = [m.name for m in models]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # RMSE comparison
    ax = axes[0]
    x = np.arange(len(model_names))
    width = 0.25
    
    for i, action in enumerate(ACTION_LABELS):
        rmse_values = [m.metrics[f'{action}_rmse'] for m in models]
        ax.bar(x + i*width, rmse_values, width, label=action.capitalize())
    
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('RMSE (lower is better)', fontweight='bold')
    ax.set_title('Model Performance - RMSE', fontsize=14, fontweight='bold')
    ax.set_xticks(x + width)
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # R² comparison
    ax = axes[1]
    r2_values = [m.metrics['overall_r2'] for m in models]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(models)]
    bars = ax.bar(model_names, r2_values, color=colors)
    
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_ylabel('R² Score (higher is better)', fontweight='bold')
    ax.set_title('Model Performance - Overall R²', fontsize=14, fontweight='bold')
    ax.set_ylim([0, 1])
    ax.set_xticklabels(model_names, rotation=15, ha='right')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.3f}',
               ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    
    filename = output_dir / "model_comparison.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"  ✅ Saved: {filename}")

def generate_training_report(models, output_dir: Path):
    """Generate text report of training results"""
    
    report_path = output_dir / "training_report.txt"
    
    with open(report_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("TELEMETRY ML TRAINING REPORT\n")
        f.write("="*70 + "\n")
        f.write(f"\nPolicy Features: {len(POLICY_FEATURES)}\n")
        f.write("No action leakage - actions excluded from state\n")
        f.write(f"\nAction Labels: {ACTION_LABELS}\n")
        f.write("="*70 + "\n")
        
        for model in models:
            f.write(f"\n{model.name}\n")
            f.write("-" * 70 + "\n")
            f.write(f"  Gas RMSE:       {model.metrics['gas_rmse']:.4f}\n")
            f.write(f"  Gas R²:         {model.metrics['gas_r2']:.4f}\n")
            f.write(f"  Brake RMSE:     {model.metrics['brake_rmse']:.4f}\n")
            f.write(f"  Brake R²:       {model.metrics['brake_r2']:.4f}\n")
            f.write(f"  Steering RMSE:  {model.metrics['steer_rmse']:.4f}\n")
            f.write(f"  Steering R²:    {model.metrics['steer_r2']:.4f}\n")
            f.write(f"  Overall RMSE:   {model.metrics['overall_rmse']:.4f}\n")
            f.write(f"  Overall R²:     {model.metrics['overall_r2']:.4f}\n")
        
        best_model = max(models, key=lambda m: m.metrics['overall_r2'])
        f.write("\n" + "="*70 + "\n")
        f.write(f"Best Model: {best_model.name}\n")
        f.write(f"Overall R²: {best_model.metrics['overall_r2']:.4f}\n")
        f.write("="*70 + "\n")
    
    print(f"  ✅ Saved: {report_path}")

def save_models(models, output_dir: Path):
    """Save trained models to disk"""
    print("\n💾 Saving trained models...")
    
    for model in models:
        # Clean model name for filename
        model_name = model.name.replace(' ', '_').replace('(', '').replace(')', '').replace(',', '').lower()
        
        # Save each action model separately
        joblib.dump(model.model_gas, output_dir / f"{model_name}_gas.pkl")
        joblib.dump(model.model_brake, output_dir / f"{model_name}_brake.pkl")
        joblib.dump(model.model_steer, output_dir / f"{model_name}_steer.pkl")
        
        print(f"  ✅ Saved: {model_name}_*.pkl")
    
    print(f"  📁 Models saved to: {output_dir}")

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Train baseline ML models for control prediction (Feature Schema Validated)'
    )
    parser.add_argument('--data_dir', type=str, default='ml_data',
                       help='Directory containing preprocessed data')
    parser.add_argument('--output_dir', type=str, default='ml_results',
                       help='Directory to save results')
    parser.add_argument('--test_size', type=float, default=0.2,
                       help='Fraction of data for testing')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    if not data_dir.exists():
        print(f"❌ Data directory not found: {data_dir}")
        print("   Run telemetry_analysis.py with --ml flag first")
        return
    
    output_dir.mkdir(exist_ok=True)
    
    # Train models
    models, X_test, y_test = train_and_evaluate_models(data_dir, args.test_size)
    
    # Generate visualizations
    plot_predictions(models, X_test, y_test, output_dir)
    plot_model_comparison(models, output_dir)
    generate_training_report(models, output_dir)
    
    save_models(models, output_dir)
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE")
    print(f"📁 Results saved to: {output_dir}")
    print("="*70)
    print("\nNote: All models trained using feature schema with no action leakage")

if __name__ == "__main__":
    main()

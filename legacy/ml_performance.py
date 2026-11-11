#!/usr/bin/env python3
"""
Model Analysis Tool for AC Telemetry ML Results
Answers key experimental questions:
1. MLP: Does capacity help or cause overfitting?
2. GA: How does mutation rate affect evolution?
3. DQL: Does reward shaping dominate policy?
"""

import joblib
import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import json
from datetime import datetime

# GA Predictor class needed for unpickling
class GAPredictor:
    """Wrapper for GA-evolved neural network weights"""
    def __init__(self, weights, input_size=11, hidden_size=32, output_size=3):
        self.weights = weights
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        # Parse weights into layers
        w1_size = input_size * hidden_size
        b1_size = hidden_size
        w2_size = hidden_size * output_size
        b2_size = output_size

        self.W1 = weights[:w1_size].reshape(input_size, hidden_size)
        self.b1 = weights[w1_size:w1_size + b1_size]
        self.W2 = weights[w1_size + b1_size:w1_size + b1_size + w2_size].reshape(hidden_size, output_size)
        self.b2 = weights[w1_size + b1_size + w2_size:w1_size + b1_size + w2_size + b2_size]

    def predict(self, X):
        """Forward pass through network"""
        # Hidden layer with ReLU
        hidden = np.maximum(0, X @ self.W1 + self.b1)
        # Output layer with tanh
        output = np.tanh(hidden @ self.W2 + self.b2)
        return output

class ModelAnalyzer:
    def __init__(self, results_dir: Path = Path("ml_results"), data_dir: Path = Path("ml_data")):
        self.results_dir = results_dir
        self.data_dir = data_dir
        self.models = {}
        self.analysis = {
            'mlp_capacity': {},
            'ga_evolution': {},
            'dql_rewards': {},
            'comparison': {}
        }

    def load_all_models(self):
        """Discover and load all models"""
        print("🔍 Discovering models...")

        # Linear Regression (baseline)
        if (self.results_dir / "linear_regression_gas.pkl").exists():
            self.models['linear'] = {
                'gas': joblib.load(self.results_dir / "linear_regression_gas.pkl"),
                'brake': joblib.load(self.results_dir / "linear_regression_brake.pkl"),
                'steer': joblib.load(self.results_dir / "linear_regression_steer.pkl"),
                'type': 'supervised'
            }
            print("  ✅ Linear Regression (baseline)")

        # MLP variants
        mlp_variants = ['64_32', '128_64_32', '256_128_64']
        for variant in mlp_variants:
            gas_path = self.results_dir / f"mlp_{variant}_gas.pkl"
            if gas_path.exists():
                self.models[f'mlp_{variant}'] = {
                    'gas': joblib.load(self.results_dir / f"mlp_{variant}_gas.pkl"),
                    'brake': joblib.load(self.results_dir / f"mlp_{variant}_brake.pkl"),
                    'steer': joblib.load(self.results_dir / f"mlp_{variant}_steer.pkl"),
                    'type': 'supervised',
                    'architecture': variant
                }
                print(f"  ✅ MLP ({variant})")

        # Genetic Algorithm
        ga_model_path = self.results_dir / "ga_model.pkl"
        ga_weights_path = self.results_dir / "ga_weights.npy"
        if ga_model_path.exists():
            ga_predictor = joblib.load(ga_model_path)
            self.models['ga'] = {
                'model': ga_predictor,
                'weights': ga_predictor.weights,
                'type': 'evolutionary'
            }
            print("  ✅ Genetic Algorithm")
        elif ga_weights_path.exists():
            self.models['ga'] = {
                'weights': np.load(ga_weights_path),
                'type': 'evolutionary'
            }
            print("  ✅ Genetic Algorithm (weights only)")

        # DQL
        dql_path = self.results_dir / "dql_model.pt"
        if dql_path.exists():
            self.models['dql'] = torch.load(dql_path, map_location='cpu')
            self.models['dql']['type'] = 'reinforcement'
            print("  ✅ Deep Q-Learning")

        print(f"\n📊 Total: {len(self.models)} models loaded")
        return self.models

    # ========================================================================
    # EXPERIMENT 1: MLP CAPACITY - OVERFITTING VS GENERALIZATION
    # ========================================================================

    def analyze_mlp_capacity(self):
        """
        Question: Does more capacity help, or does it overfit instantly?
        Metrics: Parameter count, training performance, generalization gap
        """
        print("\n" + "="*70)
        print("🧠 EXPERIMENT 1: MLP CAPACITY ANALYSIS")
        print("="*70)

        mlp_models = {k: v for k, v in self.models.items() if k.startswith('mlp_')}

        if not mlp_models:
            print("❌ No MLP models found")
            return

        results = {}

        for model_name, model_data in mlp_models.items():
            arch = model_data['architecture']
            print(f"\n🔹 MLP ({arch})")

            # 1. Count parameters
            gas_model = model_data['gas']
            total_params = 0
            layer_sizes = []

            if hasattr(gas_model, 'coefs_'):
                for coef in gas_model.coefs_:
                    total_params += coef.size
                    layer_sizes.append(coef.shape)
                for intercept in gas_model.intercepts_:
                    total_params += intercept.size

            total_params *= 3  # 3 separate models

            print(f"   Parameters: {total_params:,}")
            print(f"   Architecture: {' → '.join([str(s[0]) for s in layer_sizes])} → {layer_sizes[-1][1]}")

            # 2. Analyze loss curves (if available)
            loss_curve = getattr(gas_model, 'loss_curve_', None)

            analysis = {
                'architecture': arch,
                'parameters': total_params,
                'layer_sizes': layer_sizes,
                'converged': getattr(gas_model, 'n_iter_', 0) < getattr(gas_model, 'max_iter', 200),
                'final_loss': loss_curve[-1] if loss_curve else None
            }

            if loss_curve and len(loss_curve) > 10:
                # Check for overfitting signs
                early_loss = np.mean(loss_curve[:10])
                late_loss = np.mean(loss_curve[-10:])
                improvement = (early_loss - late_loss) / early_loss * 100

                print(f"   Training iterations: {len(loss_curve)}")
                print(f"   Loss improvement: {improvement:.1f}%")
                print(f"   Final loss: {late_loss:.4f}")

                # Overfitting indicator: loss plateaus or increases
                if len(loss_curve) > 50:
                    mid_loss = np.mean(loss_curve[25:35])
                    if late_loss > mid_loss * 0.95:  # Less than 5% improvement
                        print(f"   ⚠️  Potential plateau detected")
                        analysis['plateau'] = True

                analysis['loss_curve'] = loss_curve
                analysis['improvement_pct'] = improvement

            # 3. Weight statistics
            all_weights = []
            for output in ['gas', 'brake', 'steer']:
                m = model_data[output]
                if hasattr(m, 'coefs_'):
                    for coef in m.coefs_:
                        all_weights.append(coef.flatten())

            if all_weights:
                all_weights = np.concatenate(all_weights)
                weight_std = np.std(all_weights)
                weight_range = np.max(all_weights) - np.min(all_weights)

                print(f"   Weight std: {weight_std:.4f}")
                print(f"   Weight range: {weight_range:.4f}")

                analysis['weight_stats'] = {
                    'mean': float(np.mean(all_weights)),
                    'std': float(weight_std),
                    'min': float(np.min(all_weights)),
                    'max': float(np.max(all_weights))
                }

            results[model_name] = analysis

        # 4. Compare across architectures
        print("\n📊 CAPACITY COMPARISON:")
        print(f"{'Architecture':<20} {'Params':<12} {'Loss':<12} {'Converged'}")
        print("-" * 60)

        for model_name in sorted(results.keys(), key=lambda x: results[x]['parameters']):
            r = results[model_name]
            loss_str = f"{r['final_loss']:.4f}" if r['final_loss'] else "N/A"
            conv_str = "✓" if r['converged'] else "✗ (max iter)"
            print(f"{r['architecture']:<20} {r['parameters']:<12,} {loss_str:<12} {conv_str}")

        # Key insight
        print("\n💡 KEY INSIGHT:")
        param_counts = [r['parameters'] for r in results.values()]
        losses = [r['final_loss'] for r in results.values() if r['final_loss']]

        if len(losses) >= 2:
            if losses[-1] < losses[0]:  # Largest model has lowest loss
                print("   ✅ Larger capacity → Lower training loss (no immediate overfitting)")
            else:
                print("   ⚠️  Larger capacity → Higher training loss (possible underfitting)")

        self.analysis['mlp_capacity'] = results
        return results

    # ========================================================================
    # EXPERIMENT 2: GA MUTATION RATE - EXPLORATION VS CHAOS
    # ========================================================================

    def analyze_ga_evolution(self):
        """
        Question: When does exploration become instability?
        Metrics: Fitness evolution, population diversity, mutation sensitivity
        """
        print("\n" + "="*70)
        print("🧬 EXPERIMENT 2: GENETIC ALGORITHM EVOLUTION")
        print("="*70)

        if 'ga' not in self.models:
            print("❌ No GA model found")
            return

        ga_data = self.models['ga']

        # Load fitness history if available
        fitness_file = self.results_dir / "ga_fitness_history.npy"

        analysis = {
            'genome_size': None,
            'mutation_rate': None,
            'fitness_evolution': None
        }

        if 'weights' in ga_data:
            weights = ga_data['weights']
            analysis['genome_size'] = len(weights)

            print(f"\n🔹 Genome Analysis:")
            print(f"   Genome size: {len(weights)}")
            print(f"   Weight mean: {np.mean(weights):.4f}")
            print(f"   Weight std: {np.std(weights):.4f}")
            print(f"   Weight range: [{np.min(weights):.4f}, {np.max(weights):.4f}]")

            # Check for diversity
            unique_ratio = len(np.unique(np.round(weights, 4))) / len(weights)
            print(f"   Diversity: {unique_ratio:.2%} unique values")

            analysis['weight_stats'] = {
                'mean': float(np.mean(weights)),
                'std': float(np.std(weights)),
                'diversity': float(unique_ratio)
            }

        # Fitness evolution
        if fitness_file.exists():
            fitness_history = np.load(fitness_file)

            print(f"\n🔹 Evolution Analysis:")
            print(f"   Generations: {len(fitness_history)}")
            print(f"   Initial fitness: {fitness_history[0]:.4f}")
            print(f"   Final fitness: {fitness_history[-1]:.4f}")
            print(f"   Improvement: {((fitness_history[-1] - fitness_history[0]) / abs(fitness_history[0]) * 100):.1f}%")

            # Convergence analysis
            if len(fitness_history) > 10:
                late_std = np.std(fitness_history[-10:])
                early_std = np.std(fitness_history[:10])

                print(f"   Early volatility: {early_std:.4f}")
                print(f"   Late volatility: {late_std:.4f}")

                if late_std < early_std * 0.1:
                    print(f"   ✅ Converged (stable)")
                elif late_std > early_std:
                    print(f"   ⚠️  High volatility (possible chaos)")

                analysis['fitness_evolution'] = {
                    'history': fitness_history.tolist(),
                    'initial': float(fitness_history[0]),
                    'final': float(fitness_history[-1]),
                    'early_std': float(early_std),
                    'late_std': float(late_std),
                    'converged': late_std < early_std * 0.1
                }

        # Mutation rate inference
        print(f"\n💡 MUTATION RATE INSIGHTS:")
        if analysis.get('weight_stats', {}).get('std', 0) > 1.0:
            print("   High weight variance → Likely high mutation rate")
            print("   Risk: Unstable policies, erratic behavior")
        elif analysis.get('weight_stats', {}).get('std', 0) < 0.1:
            print("   Low weight variance → Likely low mutation rate")
            print("   Risk: Premature convergence, local optima")
        else:
            print("   Moderate weight variance → Balanced exploration")

        self.analysis['ga_evolution'] = analysis
        return analysis

    # ========================================================================
    # EXPERIMENT 3: DQL REWARD WEIGHTING - REWARD DOMINANCE TEST
    # ========================================================================

    def analyze_dql_rewards(self):
        """
        Question: When does reward shaping collapse the policy?
        Metrics: Q-values, reward components, policy stability
        """
        print("\n" + "="*70)
        print("🎮 EXPERIMENT 3: DQL REWARD ANALYSIS")
        print("="*70)

        if 'dql' not in self.models:
            print("❌ No DQL model found")
            return

        dql_data = self.models['dql']

        analysis = {
            'architecture': None,
            'training_stats': None,
            'q_values': None
        }

        # Model architecture
        if 'model_state_dict' in dql_data:
            state_dict = dql_data['model_state_dict']
            total_params = sum(p.numel() for p in state_dict.values())

            print(f"\n🔹 Network Architecture:")
            print(f"   Total parameters: {total_params:,}")

            # Layer sizes
            for name, param in state_dict.items():
                if 'weight' in name:
                    print(f"   {name}: {list(param.shape)}")

            analysis['architecture'] = {
                'parameters': total_params,
                'layers': {k: list(v.shape) for k, v in state_dict.items() if 'weight' in k}
            }

        # Training history
        if 'loss_history' in dql_data:
            losses = dql_data['loss_history']

            print(f"\n🔹 Training History:")
            print(f"   Episodes/steps: {len(losses)}")
            print(f"   Initial loss: {losses[0]:.4f}")
            print(f"   Final loss: {losses[-1]:.4f}")

            # Stability check
            if len(losses) > 100:
                recent_std = np.std(losses[-100:])
                print(f"   Recent volatility: {recent_std:.4f}")

                if recent_std > 1.0:
                    print(f"   ⚠️  High loss variance → Unstable training")

                analysis['training_stats'] = {
                    'total_steps': len(losses),
                    'initial_loss': float(losses[0]),
                    'final_loss': float(losses[-1]),
                    'recent_volatility': float(recent_std)
                }

        # Reward component analysis (if logged)
        reward_log = self.results_dir / "dql_reward_breakdown.npy"
        if reward_log.exists():
            rewards = np.load(reward_log, allow_pickle=True).item()

            print(f"\n🔹 Reward Components:")
            for component, values in rewards.items():
                print(f"   {component}: mean={np.mean(values):.3f}, std={np.std(values):.3f}")

            analysis['reward_breakdown'] = {
                k: {'mean': float(np.mean(v)), 'std': float(np.std(v))}
                for k, v in rewards.items()
            }

        # Q-value statistics (if available)
        if 'q_values' in dql_data:
            q_vals = dql_data['q_values']

            print(f"\n🔹 Q-Value Analysis:")
            print(f"   Mean Q-value: {np.mean(q_vals):.3f}")
            print(f"   Q-value range: [{np.min(q_vals):.3f}, {np.max(q_vals):.3f}]")

            analysis['q_values'] = {
                'mean': float(np.mean(q_vals)),
                'std': float(np.std(q_vals)),
                'min': float(np.min(q_vals)),
                'max': float(np.max(q_vals))
            }

        print(f"\n💡 REWARD SHAPING INSIGHTS:")
        print("   Check if one reward component dominates:")
        print("   - Speed-heavy → Wall riding, aggressive")
        print("   - Safety-heavy → Overcautious, stalling")
        print("   - Balanced → Smooth trajectory")

        self.analysis['dql_rewards'] = analysis
        return analysis

    # ========================================================================
    # CROSS-MODEL COMPARISON
    # ========================================================================

    def compare_all_models(self):
        """Compare models across all metrics"""
        print("\n" + "="*70)
        print("📊 CROSS-MODEL COMPARISON")
        print("="*70)

        comparison = {
            'supervised': [],
            'evolutionary': [],
            'reinforcement': []
        }

        # Categorize models
        for name, data in self.models.items():
            model_type = data.get('type', 'unknown')

            model_info = {
                'name': name,
                'type': model_type,
                'parameters': 0
            }

            # Count parameters
            if name == 'linear':
                model_info['parameters'] = len(data['gas'].coef_) * 3 + 3
            elif name.startswith('mlp_'):
                params = 0
                if hasattr(data['gas'], 'coefs_'):
                    for coef in data['gas'].coefs_:
                        params += coef.size
                    for intercept in data['gas'].intercepts_:
                        params += intercept.size
                model_info['parameters'] = params * 3
            elif name == 'ga' and 'weights' in data:
                model_info['parameters'] = len(data['weights'])
            elif name == 'dql' and 'model_state_dict' in data:
                model_info['parameters'] = sum(p.numel() for p in data['model_state_dict'].values())

            if model_type in comparison:
                comparison[model_type].append(model_info)

        # Print summary
        print(f"\n🔹 Model Categories:")
        for category, models in comparison.items():
            if models:
                print(f"\n   {category.upper()}:")
                for m in models:
                    print(f"      {m['name']:<20} {m['parameters']:>10,} params")

        self.analysis['comparison'] = comparison
        return comparison

    # ========================================================================
    # VISUALIZATION & REPORTING
    # ========================================================================

    def visualize_experiments(self):
        """Create comprehensive visualization answering all 3 questions"""
        print("\n📊 Generating experiment visualizations...")

        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # EXPERIMENT 1: MLP Capacity
        if 'mlp_capacity' in self.analysis and self.analysis['mlp_capacity']:
            ax1 = fig.add_subplot(gs[0, :])

            mlp_data = self.analysis['mlp_capacity']
            architectures = [d['architecture'] for d in mlp_data.values()]
            params = [d['parameters'] for d in mlp_data.values()]
            losses = [d.get('final_loss', 0) for d in mlp_data.values()]

            x = np.arange(len(architectures))
            width = 0.35

            ax1_twin = ax1.twinx()
            bars1 = ax1.bar(x - width/2, params, width, label='Parameters', alpha=0.7, color='#3498db')
            bars2 = ax1_twin.bar(x + width/2, losses, width, label='Final Loss', alpha=0.7, color='#e74c3c')

            ax1.set_xlabel('MLP Architecture', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Parameter Count', fontsize=11, color='#3498db')
            ax1_twin.set_ylabel('Training Loss', fontsize=11, color='#e74c3c')
            ax1.set_title('Experiment 1: MLP Capacity vs Performance', fontsize=14, fontweight='bold')
            ax1.set_xticks(x)
            ax1.set_xticklabels(architectures)
            ax1.tick_params(axis='y', labelcolor='#3498db')
            ax1_twin.tick_params(axis='y', labelcolor='#e74c3c')
            ax1.legend(loc='upper left')
            ax1_twin.legend(loc='upper right')
            ax1.grid(alpha=0.3, axis='y')

        # EXPERIMENT 2: GA Evolution
        if 'ga_evolution' in self.analysis and self.analysis['ga_evolution'].get('fitness_evolution'):
            ax2 = fig.add_subplot(gs[1, :])

            fitness_hist = self.analysis['ga_evolution']['fitness_evolution']['history']

            ax2.plot(fitness_hist, linewidth=2, color='#2ecc71', alpha=0.8)
            ax2.fill_between(range(len(fitness_hist)), fitness_hist, alpha=0.3, color='#2ecc71')
            ax2.set_xlabel('Generation', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Fitness', fontsize=12, fontweight='bold')
            ax2.set_title('Experiment 2: GA Fitness Evolution', fontsize=14, fontweight='bold')
            ax2.grid(alpha=0.3)

            # Add convergence indicators
            early_std = self.analysis['ga_evolution']['fitness_evolution']['early_std']
            late_std = self.analysis['ga_evolution']['fitness_evolution']['late_std']

            ax2.text(0.02, 0.98, f'Early volatility: {early_std:.4f}\nLate volatility: {late_std:.4f}',
                    transform=ax2.transAxes, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
                    fontsize=10)

        # EXPERIMENT 3: DQL Training
        if 'dql_rewards' in self.analysis and self.analysis['dql_rewards'].get('training_stats'):
            ax3 = fig.add_subplot(gs[2, :])

            # Placeholder - would need actual loss history
            ax3.text(0.5, 0.5, 'DQL Loss History\n(Load from dql_model.pt["loss_history"])',
                    ha='center', va='center', fontsize=12, transform=ax3.transAxes)
            ax3.set_title('Experiment 3: DQL Training Stability', fontsize=14, fontweight='bold')

        plt.suptitle('ML Experiments: Capacity, Evolution, Rewards', fontsize=16, fontweight='bold', y=0.995)

        output_path = self.results_dir / "experiment_analysis.png"
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"✅ Saved to: {output_path}")

        plt.close()

    def generate_experiment_report(self):
        """Generate report answering the 3 experimental questions"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        report_path = self.results_dir / "EXPERIMENTAL_ANALYSIS.md"

        with open(report_path, 'w', encoding='utf-8') as f:

            f.write(f"# Experimental Analysis Report\n\n")
            f.write(f"**Generated:** {timestamp}\n\n")
            f.write(f"---\n\n")

            # Experiment 1
            f.write(f"## Experiment 1: MLP Capacity\n\n")
            f.write(f"**Question:** Does more capacity help, or does it overfit instantly?\n\n")

            if 'mlp_capacity' in self.analysis and self.analysis['mlp_capacity']:
                f.write(f"### Results\n\n")
                for name, data in self.analysis['mlp_capacity'].items():
                    f.write(f"**{data['architecture']}**\n")
                    f.write(f"- Parameters: {data['parameters']:,}\n")
                    f.write(f"- Final Loss: {data.get('final_loss', 'N/A')}\n")
                    f.write(f"- Converged: {'Yes' if data.get('converged') else 'No'}\n\n")

                f.write(f"### Answer\n\n")
                f.write(f"TODO: Fill in after test lap results\n\n")

            # Experiment 2
            f.write(f"## Experiment 2: GA Mutation Rate\n\n")
            f.write(f"**Question:** When does exploration become instability?\n\n")

            if 'ga_evolution' in self.analysis:
                f.write(f"### Results\n\n")
                if self.analysis['ga_evolution'] and 'fitness_evolution' in self.analysis['ga_evolution'] and self.analysis['ga_evolution']['fitness_evolution']:
                    evo = self.analysis['ga_evolution']['fitness_evolution']
                    f.write(f"- Fitness improvement: {evo['final'] - evo['initial']:.4f}\n")
                    f.write(f"- Convergence: {'Yes' if evo.get('converged') else 'No'}\n")
                    f.write(f"- Volatility: Early={evo['early_std']:.4f}, Late={evo['late_std']:.4f}\n\n")
                elif self.analysis['ga_evolution'] and 'weight_stats' in self.analysis['ga_evolution']:
                    stats = self.analysis['ga_evolution']['weight_stats']
                    f.write(f"- Genome size: {self.analysis['ga_evolution'].get('genome_size', 'N/A')}\n")
                    f.write(f"- Weight std: {stats['std']:.4f}\n")
                    f.write(f"- Weight diversity: {stats['diversity']:.2%}\n\n")
                    f.write(f"⚠️ No fitness history found. Run GA training to generate fitness_history.npy\n\n")
                else:
                    f.write(f"⚠️ No GA analysis data available\n\n")

                f.write(f"### Answer\n\n")
                f.write(f"TODO: Run GA with different mutation rates (0.02, 0.05, 0.10)\n\n")

            # Experiment 3
            f.write(f"## Experiment 3: DQL Reward Weighting\n\n")
            f.write(f"**Question:** When does reward shaping collapse the policy?\n\n")

            if 'dql_rewards' in self.analysis:
                f.write(f"### Results\n\n")
                if self.analysis['dql_rewards'] and 'training_stats' in self.analysis['dql_rewards'] and self.analysis['dql_rewards']['training_stats']:
                    stats = self.analysis['dql_rewards']['training_stats']
                    f.write(f"- Training steps: {stats['total_steps']}\n")
                    f.write(f"- Loss change: {stats['initial_loss']:.4f} → {stats['final_loss']:.4f}\n")
                    f.write(f"- Stability: {stats.get('recent_volatility', 'N/A')}\n\n")
                elif self.analysis['dql_rewards'] and 'architecture' in self.analysis['dql_rewards'] and self.analysis['dql_rewards']['architecture']:
                    arch = self.analysis['dql_rewards']['architecture']
                    f.write(f"- Network parameters: {arch['parameters']:,}\n")
                    f.write(f"- Layers: {len(arch['layers'])}\n\n")
                    f.write(f"⚠️ No training history found in model file\n\n")
                else:
                    f.write(f"⚠️ No DQL analysis data available\n\n")

                f.write(f"### Answer\n\n")
                f.write(f"TODO: Test different reward weights and observe behavior\n\n")

            f.write(f"---\n\n")
            f.write(f"**Next Steps:**\n")
            f.write(f"1. Run all test laps\n")
            f.write(f"2. Record completion %, crashes, failure modes\n")
            f.write(f"3. Compare training metrics vs real performance\n")

        print(f"✅ Experiment report: {report_path}")

        # JSON version
        json_path = self.results_dir / "experiment_analysis.json"
        with open(json_path, 'w', encoding='utf-8') as f:

            json.dump(self.analysis, f, indent=2)
        print(f"✅ JSON data: {json_path}")

def main():
    print("🔬 AC TELEMETRY - EXPERIMENTAL ANALYSIS")
    print("="*70)
    print("Answering:")
    print("  1. MLP: Capacity vs Overfitting")
    print("  2. GA: Mutation Rate Sensitivity")
    print("  3. DQL: Reward Weight Dominance")
    print("="*70)

    analyzer = ModelAnalyzer()
    analyzer.load_all_models()

    # Run experimental analyses
    analyzer.analyze_mlp_capacity()
    analyzer.analyze_ga_evolution()
    analyzer.analyze_dql_rewards()
    analyzer.compare_all_models()

    # Generate outputs
    analyzer.visualize_experiments()
    analyzer.generate_experiment_report()

    print("\n" + "="*70)
    print("✅ ANALYSIS COMPLETE")
    print("="*70)
    print("\nGenerated:")
    print("  📊 experiment_analysis.png")
    print("  📄 EXPERIMENTAL_ANALYSIS.md")
    print("  📄 experiment_analysis.json")
    print("\n💡 Fill in test results to complete the analysis!")


if __name__ == "__main__":
    main()
import pandas as pd
import sys

# Remove first and last laps from the CSV file
def clean_laps(csv_file):
    df = pd.read_csv(csv_file)
    if "current_lap" not in df.columns:
        raise ValueError("CSV missing 'current_lap' column")
    min_lap, max_lap = df["current_lap"].min(), df["current_lap"].max()
    df = df[(df["current_lap"] != min_lap) & (df["current_lap"] != max_lap)]
    df.to_csv(csv_file, index=False)
    print(f"[CLEAN] {csv_file}: removed laps {min_lap} & {max_lap}")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python clean_laps.py <csv_file>")
    else:
        clean_laps(sys.argv[1])


def export_gas_steer(csv_file, output_file=None):
    """
    Export only the 'gas' and 'steer' columns from a CSV.
    Optionally save to a new CSV file.
    """
    # Read the CSV
    df = pd.read_csv(csv_file)

    # Check required columns
    required_cols = ["gas", "steer"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV missing '{col}' column")

    # Keep only gas and steer
    df_export = df[required_cols]

    # Default output filename
    if output_file is None:
        output_file = csv_file.replace(".csv", "_gas_steer.csv")

    # Save to CSV
    df_export.to_csv(output_file, index=False)
    print(f"[EXPORT] Saved gas and steer to {output_file}")

if __name__ == "__main__":
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        print("Usage: python export_gas_steer.py <csv_file> [output_file]")
    else:
        csv_file = sys.argv[1]
        output_file = sys.argv[2] if len(sys.argv) == 3 else None
        export_gas_steer(csv_file, output_file)

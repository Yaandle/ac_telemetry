import argparse
import pandas as pd


def _default_output(csv_file, suffix):
    stem = csv_file[:-4] if csv_file.lower().endswith(".csv") else csv_file
    return f"{stem}_{suffix}.csv"


# Remove first and last laps (out-lap and partial in-lap) into a new CSV
def clean_laps(csv_file, output_file=None):
    df = pd.read_csv(csv_file)
    if "current_lap" not in df.columns:
        raise ValueError("CSV missing 'current_lap' column")
    min_lap, max_lap = df["current_lap"].min(), df["current_lap"].max()
    df = df[(df["current_lap"] != min_lap) & (df["current_lap"] != max_lap)]

    output_file = output_file or _default_output(csv_file, "clean")
    if output_file == csv_file:
        raise ValueError("Refusing to overwrite the source CSV; pass a different output file")
    df.to_csv(output_file, index=False)
    print(f"[CLEAN] {csv_file}: removed laps {min_lap} & {max_lap} -> {output_file}")
    return output_file


def export_gas_steer(csv_file, output_file=None):
    """
    Export only the 'gas' and 'steer' columns from a CSV into a new CSV.
    """
    df = pd.read_csv(csv_file)

    required_cols = ["gas", "steer"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"CSV missing '{col}' column")

    output_file = output_file or _default_output(csv_file, "gas_steer")
    if output_file == csv_file:
        raise ValueError("Refusing to overwrite the source CSV; pass a different output file")
    df[required_cols].to_csv(output_file, index=False)
    print(f"[EXPORT] Saved gas and steer to {output_file}")
    return output_file


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export subsets of a telemetry CSV. The source file is never modified.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name, help_text in (("clean", "drop the first and last laps"),
                            ("gas-steer", "keep only the gas and steer columns")):
        p = sub.add_parser(name, help=help_text)
        p.add_argument("csv_file")
        p.add_argument("output_file", nargs="?")
    args = parser.parse_args()

    if args.command == "clean":
        clean_laps(args.csv_file, args.output_file)
    else:
        export_gas_steer(args.csv_file, args.output_file)

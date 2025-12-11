import duckdb
import pandas as pd

csv_path = "     "      # Add path to your data
con = duckdb.connect()

# Fetch all laps without row_num
query = f"""
SELECT 
    current_lap,
    CAST(gas AS DOUBLE) AS gas,
    CAST(brake AS DOUBLE) AS brake
FROM read_csv_auto('{csv_path}', header=True)
ORDER BY current_lap, lap_time_sec
"""
df = con.execute(query).fetchdf()

# Optional: round for readability
df['gas'] = df['gas'].round(3)
df['brake'] = df['brake'].round(3)

# Export to CSV 
df.to_csv("lap_gas_brake.csv", index=False)
con.close()

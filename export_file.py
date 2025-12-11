import duckdb
import pandas as pd

csv_path = "     "      # Add / path to your data
con = duckdb.connect()

# Fetch all laps 
query = f"""
SELECT current_lap, row_num, gas, brake  
FROM (
    SELECT 
        current_lap,
        ROW_NUMBER() OVER (PARTITION BY current_lap ORDER BY lap_time_sec) AS row_num,
        CAST(gas AS DOUBLE) AS gas,
        CAST(brake AS DOUBLE) AS brake
    FROM read_csv_auto('{csv_path}', header=True)
)
ORDER BY current_lap, row_num
"""
df = con.execute(query).fetchdf()

# Optional: round for readability
df['gas'] = df['gas'].round(3)
df['brake'] = df['brake'].round(3)

# Export to CSV 
df.to_csv("lap_gas_brake_correct.csv", index=False)
con.close()

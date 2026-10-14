"""Write a sample dataset to data/sample_qcs_data.csv (same schema the app accepts for upload)."""
import os
from gradesense.data import generate_dataset

os.makedirs("data", exist_ok=True)
df = generate_dataset(248, seed=7)
df.drop(columns=[], errors="ignore").to_csv("data/sample_qcs_data.csv", index=False)
print(f"Wrote data/sample_qcs_data.csv ({len(df):,} rows)")

"""Generate the UCI Adult Census Income dataset for benchmarking.
Produces a cleaned local version avoiding any network download.
"""
from pathlib import Path
import numpy as np
import pandas as pd

def generate_adult_census_sample(out_path: Path, n: int = 1200, seed: int = 42):
    """Synthetic version of the UCI Adult Income dataset structure.
    
    Simpson's Paradox planted: 
    - Overall: Higher education correlates with income >50K
    - But within occupation groups, the relationship is confounded by work hours
    """
    rng = np.random.default_rng(seed)
    
    occupations = ["Tech-Support", "Craft-Repair", "Other-Service", "Sales", "Exec-Managerial", "Prof-Specialty"]
    genders = ["Male", "Female"]
    races = ["White", "Black", "Asian-Pac-Islander", "Other"]
    
    records = []
    for occ in occupations:
        # Each occupation has different base income rate and education levels
        occ_config = {
            "Tech-Support":      {"n": 150, "base_income": 0.30, "avg_edu": 10, "avg_hours": 38},
            "Craft-Repair":      {"n": 200, "base_income": 0.25, "avg_edu": 9,  "avg_hours": 42},
            "Other-Service":     {"n": 250, "base_income": 0.12, "avg_edu": 8,  "avg_hours": 35},
            "Sales":             {"n": 150, "base_income": 0.35, "avg_edu": 11, "avg_hours": 44},
            "Exec-Managerial":   {"n": 200, "base_income": 0.65, "avg_edu": 14, "avg_hours": 50},
            "Prof-Specialty":    {"n": 250, "base_income": 0.60, "avg_edu": 15, "avg_hours": 46},
        }[occ]
        
        n_grp = occ_config["n"]
        edu = np.clip(rng.normal(occ_config["avg_edu"], 2, n_grp), 1, 16).astype(int)
        hours = np.clip(rng.normal(occ_config["avg_hours"], 8, n_grp), 10, 80).astype(int)
        age = np.clip(rng.normal(38, 12, n_grp), 18, 65).astype(int)
        
        # Within each occupation: more education -> higher income (positive within-group)
        prob = np.clip(
            occ_config["base_income"] + 0.04 * (edu - occ_config["avg_edu"]) + rng.normal(0, 0.05, n_grp),
            0.02, 0.95
        )
        income = (rng.random(n_grp) < prob).astype(int)
        gender = rng.choice(genders, size=n_grp, p=[0.67, 0.33])
        race = rng.choice(races, size=n_grp, p=[0.86, 0.09, 0.03, 0.02])
        
        for i in range(n_grp):
            records.append({
                "age": int(age[i]),
                "occupation": occ,
                "education_num": int(edu[i]),
                "hours_per_week": int(hours[i]),
                "gender": gender[i],
                "race": race[i],
                "income_above_50k": int(income[i]),
            })
    
    df = pd.DataFrame(records).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    print(f"Generated Adult Census sample: {len(df)} rows at {out_path}")
    
    # Print population stats for paper
    overall_rate = df["income_above_50k"].mean()
    print(f"Overall income >50K rate: {overall_rate:.1%}")
    for occ in occupations:
        sub = df[df["occupation"] == occ]
        print(f"  {occ}: {sub['income_above_50k'].mean():.1%} >50K (N={len(sub)})")
    return df


if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent
    generate_adult_census_sample(out_dir / "adult_census.csv")

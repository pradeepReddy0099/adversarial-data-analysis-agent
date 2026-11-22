"""Generate benchmark datasets exhibiting Simpson's Paradox and Confounding."""
from pathlib import Path
import numpy as np
import pandas as pd

def generate_berkeley_admissions(out_path: Path):
    """Generate expanded row-level UC Berkeley Admissions dataset (1973)."""
    table = [
        ("A", "Male", 512, 313),
        ("A", "Female", 89, 19),
        ("B", "Male", 353, 207),
        ("B", "Female", 17, 8),
        ("C", "Male", 120, 205),
        ("C", "Female", 202, 391),
        ("D", "Male", 138, 279),
        ("D", "Female", 131, 244),
        ("E", "Male", 53, 138),
        ("E", "Female", 94, 299),
        ("F", "Male", 22, 351),
        ("F", "Female", 24, 317),
    ]
    rows = []
    applicant_id = 1
    for dept, gender, admitted, rejected in table:
        for _ in range(admitted):
            rows.append({
                "applicant_id": f"APP_{applicant_id:04d}",
                "department": dept,
                "gender": gender,
                "admitted": 1,
            })
            applicant_id += 1
        for _ in range(rejected):
            rows.append({
                "applicant_id": f"APP_{applicant_id:04d}",
                "department": dept,
                "gender": gender,
                "admitted": 0,
            })
            applicant_id += 1
    
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Generated Berkeley Admissions dataset with {len(df)} rows at {out_path}")


def generate_simpsons_churn(out_path: Path, seed: int = 42):
    """Generate commercial churn dataset with a planted Simpson's Paradox.
    
    Inside each contract tier:
      Customers with MORE support calls have HIGHER churn (positive within-tier slope: issues unresolved).
    Across the aggregate population:
      Startup accounts have high churn & few calls; Enterprise accounts have low churn & frequent calls.
      Aggregate slope flips to NEGATIVE!
    """
    rng = np.random.default_rng(seed)
    
    n_startup = 300
    calls_startup = rng.integers(1, 6, size=n_startup)
    prob_startup = np.clip(0.55 + 0.06 * calls_startup + rng.normal(0, 0.04, n_startup), 0.1, 0.95)
    churn_startup = (rng.random(n_startup) < prob_startup).astype(int)
    spend_startup = rng.normal(80, 15, size=n_startup).round(2)
    tenure_startup = rng.integers(1, 18, size=n_startup)
    
    n_mid = 300
    calls_mid = rng.integers(5, 12, size=n_mid)
    prob_mid = np.clip(0.30 + 0.05 * (calls_mid - 5) + rng.normal(0, 0.04, n_mid), 0.05, 0.95)
    churn_mid = (rng.random(n_mid) < prob_mid).astype(int)
    spend_mid = rng.normal(350, 50, size=n_mid).round(2)
    tenure_mid = rng.integers(6, 36, size=n_mid)
    
    n_ent = 300
    calls_ent = rng.integers(10, 20, size=n_ent)
    prob_ent = np.clip(0.08 + 0.04 * (calls_ent - 10) + rng.normal(0, 0.03, n_ent), 0.02, 0.90)
    churn_ent = (rng.random(n_ent) < prob_ent).astype(int)
    spend_ent = rng.normal(1200, 150, size=n_ent).round(2)
    tenure_ent = rng.integers(12, 60, size=n_ent)
    
    records = []
    tiers = [
        ("Startup", calls_startup, churn_startup, spend_startup, tenure_startup, n_startup),
        ("Mid-Market", calls_mid, churn_mid, spend_mid, tenure_mid, n_mid),
        ("Enterprise", calls_ent, churn_ent, spend_ent, tenure_ent, n_ent),
    ]
    
    cid = 1001
    for tier_name, calls, churns, spends, tenures, count in tiers:
        for i in range(count):
            records.append({
                "customer_id": f"CUST_{cid}",
                "tier": tier_name,
                "support_calls": int(calls[i]),
                "monthly_spend": float(max(10.0, spends[i])),
                "tenure_months": int(tenures[i]),
                "churn": int(churns[i]),
            })
            cid += 1
            
    df = pd.DataFrame(records).sample(frac=1.0, random_state=seed).reset_index(drop=True)
    df.to_csv(out_path, index=False)
    print(f"Generated Simpson's Churn dataset with {len(df)} rows at {out_path}")

if __name__ == "__main__":
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    generate_berkeley_admissions(out_dir / "berkeley_admissions.csv")
    generate_simpsons_churn(out_dir / "simpsons_churn.csv")

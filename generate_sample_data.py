"""Generate sample credit risk data for testing the pipeline."""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def generate_sample_data(n_samples: int = 10000, seed: int = 42) -> pd.DataFrame:
    """Generate realistic credit risk dataset."""
    np.random.seed(seed)
    
    # Date range: 2 years of applications
    start_date = datetime(2022, 1, 1)
    dates = [start_date + timedelta(days=np.random.randint(0, 730)) for _ in range(n_samples)]
    
    # Features
    data = {
        "application_id": range(1, n_samples + 1),
        "application_date": dates,
        
        # Demographics
        "age": np.random.randint(21, 75, n_samples),
        "income": np.random.lognormal(10.5, 0.5, n_samples).astype(int),
        "employment_years": np.random.exponential(5, n_samples).clip(0, 40).astype(int),
        "home_ownership": np.random.choice(["RENT", "OWN", "MORTGAGE", "OTHER"], n_samples, p=[0.4, 0.2, 0.35, 0.05]),
        
        # Credit history
        "credit_score": np.random.normal(650, 100, n_samples).clip(300, 850).astype(int),
        "num_tradelines": np.random.poisson(8, n_samples).clip(0, 50),
        "num_delinquencies": np.random.poisson(0.5, n_samples).clip(0, 10),
        "utilization_rate": np.random.beta(2, 5, n_samples).clip(0, 1),
        "years_credit_history": np.random.exponential(8, n_samples).clip(0, 50),
        
        # Loan characteristics
        "loan_amount": np.random.lognormal(9.5, 0.6, n_samples).astype(int),
        "loan_term": np.random.choice([12, 24, 36, 48, 60], n_samples, p=[0.1, 0.2, 0.3, 0.25, 0.15]),
        "interest_rate": np.random.normal(12, 4, n_samples).clip(3, 36),
        "dti_ratio": np.random.beta(2, 5, n_samples).clip(0, 0.8),
        
        # Purpose
        "loan_purpose": np.random.choice(
            ["debt_consolidation", "credit_card", "home_improvement", "major_purchase", "other"],
            n_samples, p=[0.35, 0.25, 0.15, 0.15, 0.1]
        ),
    }
    
    df = pd.DataFrame(data)
    
    # Create realistic default probability based on features
    # Higher risk: low credit score, high utilization, high DTI, delinquencies
    logit_p = (
        -3.0
        + (700 - df["credit_score"]) / 50 * 0.8
        + df["utilization_rate"] * 1.5
        + df["dti_ratio"] * 2.0
        + df["num_delinquencies"] * 0.4
        - df["employment_years"] / 10 * 0.3
        - df["income"] / 100000 * 0.2
        + np.random.normal(0, 0.5, n_samples)
    )
    
    prob_default = 1 / (1 + np.exp(-logit_p))
    df["default"] = np.random.binomial(1, prob_default)
    
    return df


def main():
    print("Generating sample credit risk data...")
    df = generate_sample_data(10000)
    
    # Save
    df.to_csv("data/raw/credit_data.csv", index=False)
    print(f"Saved {len(df)} rows to data/raw/credit_data.csv")
    print(f"Default rate: {df['default'].mean():.2%}")
    print(f"Date range: {df['application_date'].min()} to {df['application_date'].max()}")
    print(f"\nColumns: {list(df.columns)}")


if __name__ == "__main__":
    main()
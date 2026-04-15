"""
ML Model Training Script
========================
Trains a Gradient Boosting classifier on realistic synthetic Indian loan data
and saves the model to disk.

Run once before starting the FastAPI server:
    python ml/train_model.py

The model predicts the probability of a loan applicant being high-risk,
which is blended (40 % weight) with the CIBIL-style formula score at
inference time.
"""

from __future__ import annotations

import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ── Output path ───────────────────────────────────────────────────────────────
OUTPUT_PATH = Path(__file__).parent / "loan_risk_model.pkl"


# ── Realistic Indian loan dataset ─────────────────────────────────────────────

def generate_dataset(n_samples: int = 8000, seed: int = 42) -> pd.DataFrame:
    """
    Generate a synthetic but realistic Indian loan applicant dataset.

    Income distribution:
      - 55 % salaried lower-middle (₹10k–₹40k/month)
      - 25 % salaried middle (₹40k–₹1.5L/month)
      - 10 % self-employed / farmer (₹8k–₹60k/month, irregular)
      - 10 % students / other (₹5k–₹20k/month)

    Loan amounts follow RBI micro/retail lending patterns.
    """
    rng = np.random.default_rng(seed)

    n = n_samples

    # ── Employment segments ───────────────────────────────────────────────────
    seg = rng.choice(["sal_low", "sal_mid", "self", "student"], size=n,
                     p=[0.55, 0.25, 0.10, 0.10])

    monthly_income = np.where(
        seg == "sal_low",   rng.uniform(10_000,  40_000, n),
        np.where(
        seg == "sal_mid",   rng.uniform(40_000, 150_000, n),
        np.where(
        seg == "self",      rng.uniform(8_000,   60_000, n),
                            rng.uniform(5_000,   20_000, n),  # student
        )))

    # Employment code: salaried=0, self_employed=1, farmer=2, student=3, other=4
    emp_code = np.where(
        seg == "sal_low",  0,
        np.where(seg == "sal_mid", 0,
        np.where(seg == "self",    rng.choice([1, 2], size=n),
                                   rng.choice([3, 4], size=n))))

    # ── Expenses (correlated with income) ────────────────────────────────────
    expense_ratio = rng.beta(3, 2, n) * 0.85 + 0.15   # 15 % – 100 %
    monthly_expenses = monthly_income * expense_ratio

    # ── Existing debt ────────────────────────────────────────────────────────
    # Most borrowers have 0–2× annual income in outstanding loans
    has_loan = rng.random(n) < 0.60   # 60 % already have loans
    existing_loans = np.where(
        has_loan,
        monthly_income * 12 * rng.beta(1.5, 3, n) * 2.5,
        0.0,
    )

    # ── Current EMI ──────────────────────────────────────────────────────────
    # EMI drawn from existing_loans; students/farmers skew lower
    emi_ratio = np.where(
        has_loan,
        rng.uniform(0.05, 0.55, n),   # 5 % – 55 % of income
        0.0,
    )
    emi_amount = monthly_income * emi_ratio

    # ── Repayment history ────────────────────────────────────────────────────
    # Correlated negatively with emi_ratio (high EMI burden → worse history)
    base_hist = rng.uniform(20, 100, n)
    history_penalty = emi_ratio * 30
    repayment_score = np.clip(base_hist - history_penalty, 0, 100)

    # ── Loan requested ───────────────────────────────────────────────────────
    # Range: ₹20k micro-loan → 10× annual income
    loan_multiplier = rng.choice(
        [0.5, 1.0, 1.5, 2.0, 3.0, 5.0, 8.0],
        size=n,
        p=[0.10, 0.20, 0.25, 0.20, 0.12, 0.08, 0.05],
    )
    loan_requested = monthly_income * 12 * loan_multiplier
    loan_requested = np.clip(loan_requested, 20_000, 5_000_000)

    # ── Tenure ───────────────────────────────────────────────────────────────
    tenure_months = rng.choice(
        [6, 12, 18, 24, 36, 48, 60, 84, 120],
        size=n,
        p=[0.05, 0.15, 0.10, 0.20, 0.20, 0.10, 0.10, 0.05, 0.05],
    ).astype(float)

    # ── Derived features ─────────────────────────────────────────────────────
    dti = np.where(monthly_income > 0, existing_loans / (monthly_income * 12), 1.0)
    eti = np.where(monthly_income > 0, emi_amount / monthly_income, 1.0)
    savings_ratio = np.where(
        monthly_income > 0,
        (monthly_income - monthly_expenses - emi_amount) / monthly_income,
        -1.0,
    )
    loan_income_ratio = np.where(
        monthly_income > 0,
        loan_requested / (monthly_income * 12),
        10.0,
    )

    # ── CIBIL-style risk label ───────────────────────────────────────────────
    # Weights mirror the formula in risk_service.py
    rh_risk       = np.clip(100 - repayment_score, 0, 100) / 100
    emi_risk      = np.clip(eti * 200, 0, 100) / 100
    debt_risk     = np.clip(dti * 150, 0, 100) / 100
    savings_risk  = np.clip((0.20 - savings_ratio) * 200, 0, 100) / 100
    loan_risk     = np.clip((loan_income_ratio - 2) / 8 * 100, 0, 100) / 100

    formula_score = (
        0.35 * rh_risk
        + 0.30 * emi_risk
        + 0.15 * debt_risk
        + 0.10 * savings_risk
        + 0.10 * loan_risk
    )
    # Add small noise to prevent perfect alignment with formula
    formula_score += rng.normal(0, 0.03, n)
    label = (formula_score > 0.5).astype(int)

    df = pd.DataFrame({
        "monthly_income":         monthly_income,
        "monthly_expenses":       monthly_expenses,
        "existing_loans":         existing_loans,
        "emi_amount":             emi_amount,
        "repayment_history_score": repayment_score,
        "loan_amount_requested":  loan_requested,
        "loan_tenure_months":     tenure_months,
        "dti":                    dti,
        "eti":                    eti,
        "savings_ratio":          savings_ratio,
        "loan_income_ratio":      loan_income_ratio,
        "employment_type":        emp_code.astype(float),
        "label":                  label,
    })
    return df


# ── Training ──────────────────────────────────────────────────────────────────

def train() -> None:
    print("Generating realistic Indian loan training data…")
    df = generate_dataset(n_samples=10_000)

    print(f"  Class balance — Low/Medium risk: {(df.label == 0).sum()}, "
          f"High risk: {(df.label == 1).sum()}")

    feature_cols = [
        "monthly_income", "monthly_expenses", "existing_loans", "emi_amount",
        "repayment_history_score", "loan_amount_requested", "loan_tenure_months",
        "dti", "eti", "savings_ratio", "loan_income_ratio", "employment_type",
    ]

    X = df[feature_cols].values
    y = df["label"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("Training Gradient Boosting model…")
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.07,
            max_depth=4,
            min_samples_leaf=20,
            subsample=0.8,
            random_state=42,
        )),
    ])

    pipeline.fit(X_train, y_train)

    # ── Evaluation ────────────────────────────────────────────────────────────
    y_pred  = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    print("\n-- Classification Report ------------------------------------------")
    print(classification_report(y_test, y_pred, target_names=["Low/Medium Risk", "High Risk"]))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")

    # ── Save ──────────────────────────────────────────────────────────────────
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, OUTPUT_PATH)
    print(f"\nModel saved to: {OUTPUT_PATH}")


if __name__ == "__main__":
    train()

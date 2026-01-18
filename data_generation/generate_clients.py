from __future__ import annotations

import math
import random
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


# ---------- Utilities ----------

def weighted_choice(rng: random.Random, items: List[Tuple[str, float]]) -> str:
    keys = [k for k, _ in items]
    weights = [w for _, w in items]
    return rng.choices(keys, weights=weights, k=1)[0]


def make_client_id(i: int) -> str:
    return f"CLT-{i:06d}"


def random_signup_date(rng: random.Random, start: date, end: date) -> date:
    delta_days = (end - start).days
    return start + timedelta(days=rng.randint(0, delta_days))


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


# ---------- Config (keep in code for now; later we move to YAML) ----------

SEED = 42
N_CLIENTS = 2500

SIGNUP_START = date(2022, 1, 1)
SIGNUP_END = date(2025, 12, 31)

COUNTRY_WEIGHTS = [
    ("FR", 0.22), ("DE", 0.18), ("ES", 0.12), ("IT", 0.12),
    ("NL", 0.08), ("BE", 0.08), ("PT", 0.06), ("IE", 0.06),
    ("CH", 0.04), ("AT", 0.04),
]

INDUSTRY_WEIGHTS = [
    ("software_it", 0.18),
    ("professional_services", 0.16),
    ("ecommerce", 0.14),
    ("manufacturing", 0.12),
    ("retail", 0.10),
    ("healthcare", 0.08),
    ("finance", 0.08),
    ("hospitality", 0.07),
    ("education", 0.07),
]

SIZE_WEIGHTS = [
    ("micro", 0.35),   # 1-9
    ("small", 0.40),   # 10-49
    ("mid", 0.20),     # 50-249
    ("large", 0.05),   # 250+
]

PLAN_BY_SIZE = {
    "micro": [("starter", 0.80), ("pro", 0.19), ("enterprise", 0.01)],
    "small": [("starter", 0.55), ("pro", 0.40), ("enterprise", 0.05)],
    "mid": [("starter", 0.20), ("pro", 0.60), ("enterprise", 0.20)],
    "large": [("starter", 0.05), ("pro", 0.35), ("enterprise", 0.60)],
}

# Simple name generator (synthetic but plausible)
NAME_PREFIX = ["Nova", "Alto", "Blue", "Green", "Prime", "Vertex", "Swift", "Aurum", "Civic", "Delta", "Orion", "Lumen"]
NAME_SUFFIX = ["Solutions", "Consulting", "Industries", "Trading", "Labs", "Group", "Systems", "Partners", "Services", "Works"]


# ---------- Business logic ----------

def sample_employees(rng: random.Random, company_size: str) -> int:
    # Use log-normal-ish distributions per bracket to look realistic
    if company_size == "micro":
        return int(clamp(rng.lognormvariate(mu=math.log(4), sigma=0.45), 1, 9))
    if company_size == "small":
        return int(clamp(rng.lognormvariate(mu=math.log(22), sigma=0.40), 10, 49))
    if company_size == "mid":
        return int(clamp(rng.lognormvariate(mu=math.log(110), sigma=0.45), 50, 249))
    # large
    return int(clamp(rng.lognormvariate(mu=math.log(600), sigma=0.60), 250, 5000))


def estimate_mrr(rng: random.Random, plan: str, employees: int) -> float:
    """
    MRR depends on plan and weakly on size (seat-based feel) with noise.
    We keep it simple but coherent.
    """
    if plan == "starter":
        base = 59
        per_seat = 1.2
        cap = 199
    elif plan == "pro":
        base = 199
        per_seat = 1.8
        cap = 899
    else:  # enterprise
        base = 999
        per_seat = 2.4
        cap = 6500

    noisy = (base + per_seat * math.sqrt(employees) * 10) * rng.uniform(0.85, 1.20)
    return round(clamp(noisy, base * 0.8, cap), 2)


def churn_probability(plan: str, tenure_days: int) -> float:
    """
    Simple churn model:
    - starter churns more
    - churn decreases with tenure
    """
    if plan == "starter":
        base = 0.10
    elif plan == "pro":
        base = 0.06
    else:
        base = 0.03

    # reduce with tenure
    factor = 1.0 / (1.0 + tenure_days / 365.0)
    return clamp(base * factor, 0.005, base)


def generate_company_name(rng: random.Random) -> str:
    return f"{rng.choice(NAME_PREFIX)} {rng.choice(NAME_SUFFIX)}"


# ---------- Main ----------

def main() -> None:
    rng = random.Random(SEED)
    np.random.seed(SEED)

    today = date(2026, 1, 18)  # fixed for reproducibility in this project context

    rows: List[Dict] = []
    for i in range(1, N_CLIENTS + 1):
        client_id = make_client_id(i)
        company_name = generate_company_name(rng)

        country = weighted_choice(rng, COUNTRY_WEIGHTS)
        industry = weighted_choice(rng, INDUSTRY_WEIGHTS)
        company_size = weighted_choice(rng, SIZE_WEIGHTS)

        employees = sample_employees(rng, company_size)
        plan = weighted_choice(rng, PLAN_BY_SIZE[company_size])

        signup_date = random_signup_date(rng, SIGNUP_START, SIGNUP_END)
        tenure_days = max(1, (today - signup_date).days)

        p_churn = churn_probability(plan, tenure_days)
        is_active = rng.random() > p_churn

        mrr_eur = estimate_mrr(rng, plan, employees)

        rows.append(
            {
                "client_id": client_id,
                "company_name": company_name,
                "industry": industry,
                "country": country,
                "company_size": company_size,
                "employees": employees,
                "plan": plan,
                "signup_date": signup_date.isoformat(),
                "is_active": is_active,
                "mrr_eur": mrr_eur,
            }
        )

    df = pd.DataFrame(rows)

    # Basic sanity checks (enterprise-like)
    assert df["client_id"].is_unique
    assert df["employees"].min() >= 1
    assert set(df["plan"].unique()).issubset({"starter", "pro", "enterprise"})

    out_dir = Path(__file__).resolve().parents[1] / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "clients.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"Generated clients: {len(df)} -> {out_path}")


if __name__ == "__main__":
    main()


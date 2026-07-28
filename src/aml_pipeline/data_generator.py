# IMPORTS
import uuid     # Generates unique IDs (for transaction_id)
import random   
import numpy as np
import pandas as pd
from faker import Faker # Generates realistic fake names/companies
from dataclasses import dataclass


# Will pick randomly from these fixed lists
CURRENCIES = ["USD", "TWD", "EUR", "JPY"]
PURPOSE_CODES = ["TRADE_SETTLEMENT", "PAYROLL", "FAMILY_SUPPORT", "SERVICES", "UNSPECIFIED"]

@dataclass
class GeneratorConfig:
    n_entities: int = 50
    n_clean_transactions: int = 500
    start_date: str = "2026-01-01"
    end_date: str = "2026-06-30"
    seed: int = 42

def _random_date(start, end):
    """
    Returns one random timestamp between `start` and `end`.
    -> internal helper func
    """
    delta = end-start
    return start + pd.Timedelta(seconds=random.randint(0, int(delta.total_seconds())))

def make_entities(fake, n):
    """
    Creates pool of n fake account holders
    - Each either person or companies (chose 35% companies)
    """
    rows = [] 
    for i in range(n):
        # random.random() gives float btwn 0 -> 1
        # 35% prob. company
        is_company = random.random() < 0.35
        # assign fake names
        name = fake.company() if is_company else fake.name()
        rows.append({
            "entity_id": f"ENT{i:05d}", # e.g. ENT00007 — zero-padded so IDs sort nicely
            "name": name,
            "is_company": is_company,
        })
    
    # pd.DataFrame turns list of dicts (rows) into a table
    # - each dict becomes row, each key becomes a col
    return pd.DataFrame(rows)

def generate_clean_transactions(config):
    """
    Generates ordinary, non-fraudulent transaction
    """
    random.seed(config.seed)
    np.random.seed(config.seed)
    fake = Faker()
    Faker.seed(config.seed)

    entities = make_entities(fake, config.n_entities)
    start = pd.Timestamp(config.start_date)
    end = pd.Timestamp(config.end_date)

    rows = []
    for _ in range(config.n_clean_transactions):
        sender, receiver = entities.sample(2).to_dict("records")
        amount = round(min(np.random.lognormal(mean=8.0, sigma=1.1), 500_000), 2)
        rows.append({
            "transaction_id": f"TXN{uuid.uuid4().hex[:10].upper()}",
            "date": _random_date(start, end),
            "sender_id": sender["entity_id"], "sender_name": sender["name"],
            "receiver_id": receiver["entity_id"], "receiver_name": receiver["name"],
            "amount": amount, "currency": random.choice(CURRENCIES),
            "purpose_code": random.choice(PURPOSE_CODES),
        })
    return pd.DataFrame(rows), entities

if __name__ == "__main__":
    config = GeneratorConfig()
    txns, entities = generate_clean_transactions(config)
    print(txns[["transaction_id", "date", "sender_name", "amount"]].head())
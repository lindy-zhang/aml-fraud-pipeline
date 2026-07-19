# IMPORTS
import uuid     # Generates unique IDs (for transaction_id)
import random   
import numpy as np
import pandas as pd
from faker import Faker # Generates realistic fake names/companies

# Will pick randomly from these fixed lists
CURRENCIES = ["USD", "TWD", "EUR", "JPY"]
PURPOSE_CODES = ["TRADE_SETTLEMENT", "PAYROLL", "FAMILY_SUPPORT", "SERVICES", "UNSPECIFIED"]


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
            "entity_id": f"ENT{i:05d}" # e.g. ENT00007 — zero-padded so IDs sort nicely
            "name": name
            "is_company": is_company
        })
    
    # pd.DataFrame turns list of dicts (rows) into a table
    # - each dict becomes row, each key becomes a col
    return pd.DataFrame(rows)

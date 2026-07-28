"""
Synthetic data generator for cross-border remittance transactions,
calibrated against real FEIB transaction-monitoring output and FEIB's
own inward/outward remittance classification codes.

Each row mirrors the fields on a shui-dan (水單 / remittance advice).
On top of a base population of "clean" transactions, we inject four
laundering typologies:

    - structuring              : many transactions just under the real
                                   NT$500,000 reporting threshold
    - layering                  : funds routed through a short chain of
                                   entities, sometimes cycling back
    - fan-out/fan-in             : one entity rapidly splits funds to
                                   (or collects from) many counterparties
    - va_disguised_settlement    : virtual-asset settlement disguised as
                                   generic "professional services" (code
                                   19D) from many countries, converging
                                   on one exchange/VASP-segment entity --
                                   modeled on a real observed pattern

None of this is real client data.
"""

import uuid
import random
import numpy as np
import pandas as pd
from faker import Faker
from dataclasses import dataclass

# --- Constants ---------------------------------------------------------
# Currency mix is TWD-dominant, matching a Taiwan-domiciled bank's book.
CURRENCIES = ["TWD", "USD", "EUR", "JPY"]
CURRENCY_WEIGHTS = [0.55, 0.25, 0.10, 0.10]

COUNTRIES = [
    "Taiwan", "USA", "Japan", "Singapore", "United Kingdom", "Germany",
    "Italy", "Cayman Islands", "South Korea", "Netherlands", "Hong Kong",
]

# Real inward/outward remittance codes, pulled from FEIB's own
# classification documents. Kept to a representative subset rather than
# the full ~150-code list.
PURPOSE_CODES_GENERAL = [
    "19D",   # Professional/technical services and business receipts
    "599",   # Other transfer receipts
    "711",   # Receipts from merchanting trade
    "410",   # Inward remittance of wages and salaries
    "199",   # Other service receipts
]
PURPOSE_CODES_CORPORATE = [
    "210",   # Return of overseas direct investment
    "321",   # Financing offered by foreign equity shareholders
    "711",   # Merchanting trade
    "19D",   # Professional/technical services
]
VA_CODES = ["268", "368"]  # Purchase/sale of virtual assets (outward/inward)
MASKING_CODE = "693"  # bank-internal code: FX transferred from another
                        # domestic bank -- masks the customer's original
                        # stated purpose. A handful of clean transactions
                        # get relabeled to this, mirroring a real blind
                        # spot your mentor's analysis flagged.

# Real regulatory threshold cited in FEIB's classification docs (codes
# 801/802): accumulated settlement over NT$500,000 triggers reporting.
STRUCTURING_THRESHOLD_TWD = 500_000

# Segment proportions matching the real customer book observed in
# production output (96 sme / 84 exchange_vasp / 4 large_corporate out
# of 184 customers).
SEGMENTS = ["sme", "exchange_vasp", "large_corporate"]
SEGMENT_WEIGHTS = [0.52, 0.46, 0.02]


@dataclass
class GeneratorConfig:
    n_entities: int = 220
    n_clean_transactions: int = 4000
    n_structuring_rings: int = 6
    n_layering_chains: int = 5
    n_fan_networks: int = 4
    n_va_disguised_settlements: int = 3
    start_date: str = "2026-01-01"
    end_date: str = "2026-06-30"
    seed: int = 42


def _random_date(start, end):
    delta = end - start
    return start + pd.Timedelta(seconds=random.randint(0, int(delta.total_seconds())))


def _sample_amount_twd():
    """
    Amounts drawn from a lognormal distribution whose parameters are
    fit toward the real percentile figures from production output
    (50th ~NT$1.4M, 95th ~NT$66M). This is a rough calibration, not an
    exact fit -- real transaction-size distributions are heavier-tailed
    than a single lognormal can fully capture, especially at the very
    top percentiles (99th+), but this gets the bulk of the distribution
    in the right order of magnitude.
    """
    mu, sigma = 14.17, 2.0  # ln(1,423,485) ~= 14.17
    amt = np.random.lognormal(mean=mu, sigma=sigma)
    return round(min(amt, 200_000_000), 0)


def make_entities(fake, n):
    """
    Creates a pool of `n` fake account holders, each assigned a customer
    segment (sme / exchange_vasp / large_corporate) at realistic
    proportions, plus a home country.
    """
    rows = []
    for i in range(n):
        segment = random.choices(SEGMENTS, weights=SEGMENT_WEIGHTS)[0]
        is_company = segment in ("exchange_vasp", "large_corporate") or random.random() < 0.35
        name = fake.company() if is_company else fake.name()
        rows.append({
            "entity_id": f"ENT{i:05d}",
            "name": name,
            "is_company": is_company,
            "segment": segment,
            "country": random.choice(COUNTRIES),
        })
    return pd.DataFrame(rows)


def _purpose_code_for_segment(segment):
    """
    Exchange/VASP customers draw heavily from VA-specific codes; a
    smaller share of their traffic is coded as generic professional
    services (19D) -- the same mix that made the real CUST3260 pattern
    hard to see without cross-referencing counterparty country spread.
    """
    if segment == "exchange_vasp":
        return random.choices(
            VA_CODES + ["19D", "599"], weights=[0.35, 0.35, 0.20, 0.10]
        )[0]
    if segment == "large_corporate":
        return random.choice(PURPOSE_CODES_CORPORATE)
    return random.choice(PURPOSE_CODES_GENERAL)


def _inject_structuring(fake, entities, start, end, ring_id):
    """
    Same entity -> same counterparty, several amounts clustered just
    under the real NT$500,000 threshold, tight window. Because sender
    and receiver are fixed for the whole ring, this pattern already
    passes a "same source" check by construction -- worth remembering
    when we build the rule engine's same-source fix later.
    """
    sender, receiver = entities.sample(2).to_dict("records")
    n_txns = random.randint(3, 6)
    base_date = _random_date(start, end - pd.Timedelta(days=3))
    rows = []
    for k in range(n_txns):
        amt = STRUCTURING_THRESHOLD_TWD * random.uniform(0.90, 0.99)
        rows.append({
            "transaction_id": f"TXN{uuid.uuid4().hex[:10].upper()}",
            "date": base_date + pd.Timedelta(hours=random.randint(0, 40)),
            "sender_id": sender["entity_id"], "sender_name": sender["name"],
            "receiver_id": receiver["entity_id"], "receiver_name": receiver["name"],
            "amount": round(amt, 2), "currency": "TWD", "purpose_code": "599",
            "counterparty_country": receiver["country"],
            "label_is_injected_pattern": "STRUCTURING",
            "label_pattern_id": f"STRUCT-{ring_id}",
        })
    return rows


def _inject_layering_chain(fake, entities, start, end, chain_id):
    chain_len = random.randint(3, 5)
    chain_entities = entities.sample(chain_len).to_dict("records")
    if random.random() < 0.5:
        chain_entities.append(chain_entities[0])
    amount = _sample_amount_twd() * 3  # layering chains move larger sums
    base_date = _random_date(start, end - pd.Timedelta(days=2))
    rows = []
    for i in range(len(chain_entities) - 1):
        sender = chain_entities[i]
        receiver = chain_entities[i + 1]
        amount *= random.uniform(0.90, 0.98)
        rows.append({
            "transaction_id": f"TXN{uuid.uuid4().hex[:10].upper()}",
            "date": base_date + pd.Timedelta(hours=i * random.randint(2, 10)),
            "sender_id": sender["entity_id"], "sender_name": sender["name"],
            "receiver_id": receiver["entity_id"], "receiver_name": receiver["name"],
            "amount": round(amount, 2), "currency": "TWD", "purpose_code": "711",
            "counterparty_country": receiver["country"],
            "label_is_injected_pattern": "LAYERING",
            "label_pattern_id": f"LAYER-{chain_id}",
        })
    return rows


def _inject_fan_network(fake, entities, start, end, net_id):
    hub = entities.sample(1).to_dict("records")[0]
    spokes = entities.sample(random.randint(6, 12)).to_dict("records")
    fan_out = random.random() < 0.5
    base_date = _random_date(start, end - pd.Timedelta(days=1))
    rows = []
    for spoke in spokes:
        sender, receiver = (hub, spoke) if fan_out else (spoke, hub)
        amt = STRUCTURING_THRESHOLD_TWD * random.uniform(0.4, 0.9)
        rows.append({
            "transaction_id": f"TXN{uuid.uuid4().hex[:10].upper()}",
            "date": base_date + pd.Timedelta(minutes=random.randint(0, 2000)),
            "sender_id": sender["entity_id"], "sender_name": sender["name"],
            "receiver_id": receiver["entity_id"], "receiver_name": receiver["name"],
            "amount": round(amt, 2), "currency": "TWD", "purpose_code": "599",
            "counterparty_country": receiver["country"],
            "label_is_injected_pattern": "FAN_NETWORK",
            "label_pattern_id": f"FAN-{net_id}",
        })
    return rows


def _inject_va_disguised_settlement(fake, entities, start, end, net_id):
    """
    Modeled directly on the real CUST3260 finding: a wide, multi-country
    spread of payments coded as generic "professional/technical
    services" (19D) converging on ONE exchange/VASP-segment entity,
    plus a couple of genuinely VA-coded (268/368) transactions to the
    same counterparty -- the pattern you'd expect if virtual-asset
    settlement is being coded as something else specifically to avoid
    the VA-code bucket.
    """
    vasp_candidates = entities[entities["segment"] == "exchange_vasp"]
    hub = (vasp_candidates.sample(1) if len(vasp_candidates) else entities.sample(1)).to_dict("records")[0]

    n_senders = random.randint(8, 15)
    senders = entities[entities["entity_id"] != hub["entity_id"]].sample(n_senders).to_dict("records")
    base_date = _random_date(start, end - pd.Timedelta(days=5))

    rows = []
    for i, sender in enumerate(senders):
        # Most transactions coded as generic services -- the masking
        # layer. A couple are left genuinely VA-coded, which is the
        # detail that actually makes the pattern findable.
        is_va_coded = i < 2
        purpose = random.choice(VA_CODES) if is_va_coded else "19D"
        amt = _sample_amount_twd() * (0.3 if not is_va_coded else 1.5)
        rows.append({
            "transaction_id": f"TXN{uuid.uuid4().hex[:10].upper()}",
            "date": base_date + pd.Timedelta(hours=random.randint(0, 96)),
            "sender_id": sender["entity_id"], "sender_name": sender["name"],
            "receiver_id": hub["entity_id"], "receiver_name": hub["name"],
            "amount": round(amt, 2), "currency": "USD", "purpose_code": purpose,
            "counterparty_country": sender["country"],
            "label_is_injected_pattern": "VA_DISGUISED_SETTLEMENT",
            "label_pattern_id": f"VADISGUISE-{net_id}",
        })
    return rows


def generate_transactions(config):
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
        # A small share of otherwise-clean transactions get their
        # purpose masked under code 693 -- mirrors the real blind spot:
        # some genuine purposes are simply not visible in this data.
        if random.random() < 0.03:
            purpose = MASKING_CODE
        else:
            purpose = _purpose_code_for_segment(receiver["segment"])
        rows.append({
            "transaction_id": f"TXN{uuid.uuid4().hex[:10].upper()}",
            "date": _random_date(start, end),
            "sender_id": sender["entity_id"], "sender_name": sender["name"],
            "receiver_id": receiver["entity_id"], "receiver_name": receiver["name"],
            "amount": _sample_amount_twd(),
            "currency": random.choices(CURRENCIES, weights=CURRENCY_WEIGHTS)[0],
            "purpose_code": purpose,
            "counterparty_country": receiver["country"],
            "label_is_injected_pattern": "NONE", "label_pattern_id": None,
        })

    for i in range(config.n_structuring_rings):
        rows.extend(_inject_structuring(fake, entities, start, end, i))
    for i in range(config.n_layering_chains):
        rows.extend(_inject_layering_chain(fake, entities, start, end, i))
    for i in range(config.n_fan_networks):
        rows.extend(_inject_fan_network(fake, entities, start, end, i))
    for i in range(config.n_va_disguised_settlements):
        rows.extend(_inject_va_disguised_settlement(fake, entities, start, end, i))

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    return df, entities


if __name__ == "__main__":
    config = GeneratorConfig()
    txns, entities = generate_transactions(config)
    print(f"Generated {len(txns)} transactions across {len(entities)} entities")
    print("\nSegment distribution:")
    print(entities["segment"].value_counts())
    print("\nInjected pattern breakdown:")
    print(txns["label_is_injected_pattern"].value_counts())
    print("\nAmount percentiles (TWD-coded transactions):")
    twd = txns[txns["currency"] == "TWD"]["amount"]
    print(twd.quantile([0.5, 0.75, 0.9, 0.95, 0.99]))
    txns.to_csv("../../data/transactions.csv", index=False)
    entities.to_csv("../../data/entities.csv", index=False)
    print("\nSaved to data/transactions.csv and data/entities.csv")
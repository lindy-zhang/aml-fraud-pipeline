"""
Feature engineering: turns the raw transaction ledger into one feature
row per entity. Laundering typologies are properties of an entity's
BEHAVIOR OVER TIME, not any single transaction, so we pivot from
transaction-level rows to entity-level rows here.

Features map to specific typologies:
    - near_threshold_count / ratio   -> structuring (NT$500,000 threshold)
    - max_txns_in_24h                 -> velocity / high-frequency movement
    - avg_hold_time_hours             -> layering (funds in, funds back out fast)
    - unique_counterparties            -> fan-out/fan-in mule networks
    - max_countries_in_7d              -> VA-disguised settlement: this is a
                                          WINDOWED count, not a lifetime one.
                                          A busy, ordinary entity naturally
                                          touches many countries over months
                                          of activity, so a lifetime count
                                          doesn't discriminate at all -- what
                                          actually stands out is many
                                          DIFFERENT countries hitting one
                                          entity within a single week.
    - va_code_share / pct_19d          -> VA settlement possibly disguised
                                          as generic "professional services"
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .data_generator import STRUCTURING_THRESHOLD_TWD, VA_CODES

NEAR_THRESHOLD_BAND = 0.10  # within 10% of the reporting threshold


def build_entity_features(txns: pd.DataFrame, entities: pd.DataFrame) -> pd.DataFrame:
    txns = txns.copy()
    txns["date"] = pd.to_datetime(txns["date"])

    outflow = txns.rename(columns={
        "sender_id": "entity_id", "sender_name": "entity_name",
        "receiver_id": "counterparty_id",
    })
    outflow["role"] = "SENDER"
    inflow = txns.rename(columns={
        "receiver_id": "entity_id", "receiver_name": "entity_name",
        "sender_id": "counterparty_id",
    })
    inflow["role"] = "RECEIVER"
    long_df = pd.concat([outflow, inflow], ignore_index=True)

    feats = []
    for entity_id, grp in long_df.groupby("entity_id"):
        grp = grp.sort_values("date")
        sent = grp[grp["role"] == "SENDER"]
        recv = grp[grp["role"] == "RECEIVER"]

        n_txns = len(grp)
        span_days = max((grp["date"].max() - grp["date"].min()).total_seconds() / 86400, 1e-6)

        indexed = grp.set_index("date")
        counts_per_day = indexed.resample("1D").size()
        max_daily_count = counts_per_day.max() if len(counts_per_day) else 0

        # Windowed, not lifetime -- see module docstring for why this
        # matters. resample("7D") bins transactions into rolling
        # 7-day-wide buckets starting from the entity's first
        # transaction; nunique() counts distinct countries within each
        # bucket; .max() keeps the single busiest week.
        weekly_countries = indexed.resample("7D")["counterparty_country"].nunique()
        max_countries_7d = weekly_countries.max() if len(weekly_countries) else 0

        twd_grp = grp[grp["currency"] == "TWD"]
        near_threshold = twd_grp[
            (twd_grp["amount"] >= STRUCTURING_THRESHOLD_TWD * (1 - NEAR_THRESHOLD_BAND))
            & (twd_grp["amount"] < STRUCTURING_THRESHOLD_TWD)
        ]

        if len(recv) and len(sent):
            recv_times = recv["date"].sort_values().values
            sent_times = sent["date"].sort_values().values
            hold_times = []
            for rt in recv_times:
                later_sends = sent_times[sent_times > rt]
                if len(later_sends):
                    hold_times.append((later_sends[0] - rt) / np.timedelta64(1, "h"))
            avg_hold_time = float(np.mean(hold_times)) if hold_times else np.nan
        else:
            avg_hold_time = np.nan

        feats.append({
            "entity_id": entity_id,
            "entity_name": grp["entity_name"].iloc[0],
            "n_transactions": n_txns,
            "total_sent": sent["amount"].sum(),
            "total_received": recv["amount"].sum(),
            "avg_amount": grp["amount"].mean(),
            "std_amount": grp["amount"].std(ddof=0) if n_txns > 1 else 0.0,
            "max_amount": grp["amount"].max(),
            "unique_counterparties": grp["counterparty_id"].nunique(),
            "max_countries_in_7d": max_countries_7d,
            "txn_per_day": n_txns / span_days,
            "max_txns_in_24h": max_daily_count,
            "near_threshold_count": len(near_threshold),
            "near_threshold_ratio": len(near_threshold) / n_txns if n_txns else 0,
            "in_out_ratio": (
                recv["amount"].sum() / sent["amount"].sum()
                if sent["amount"].sum() > 0 else np.nan
            ),
            "avg_hold_time_hours": avg_hold_time,
            "va_code_share": grp["purpose_code"].isin(VA_CODES).mean(),
            "pct_19d_professional_services": grp["purpose_code"].eq("19D").mean(),
            "pct_masked_purpose": grp["purpose_code"].eq("693").mean(),
            "true_pattern": grp.loc[
                grp["label_is_injected_pattern"] != "NONE", "label_is_injected_pattern"
            ].mode().iat[0] if (grp["label_is_injected_pattern"] != "NONE").any() else "NONE",
        })

    feat_df = pd.DataFrame(feats).fillna(0)

    # segment lives on the entities table, not on individual transaction
    # rows, so it's merged in here rather than read off `grp` (which
    # only has transaction-level columns).
    feat_df = feat_df.merge(
        entities[["entity_id", "segment"]], on="entity_id", how="left"
    )

    return feat_df

if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from aml_pipeline.data_generator import GeneratorConfig, generate_transactions

    txns, entities = generate_transactions(GeneratorConfig())
    feats = build_entity_features(txns, entities)
    print(feats.sort_values("max_countries_in_7d", ascending=False)[
        ["entity_id", "segment", "max_countries_in_7d", "true_pattern"]
    ].head(6))
"""
Unsupervised anomaly detection over entity features.

Two complementary lenses, because they catch different notions of
"unusual":

    1. Isolation Forest -> a continuous anomaly score per entity. Works
       by building many random decision trees that split the feature
       space; genuine outliers get isolated (separated from everything
       else) in very few splits, since they sit far from the crowd on
       some combination of features. Good for RANKING entities.

    2. DBSCAN -> density-based clustering. Groups entities sitting in
       dense neighborhoods into clusters; anything that doesn't belong
       to a dense cluster gets labeled -1 (noise / density outlier).
       Gives a binary signal, a different notion of "unusual" than
       Isolation Forest's.

An entity flagged by BOTH methods is a stronger signal than either
alone -- that's used later as its own rule (R7 in the rule engine).

Features are standardized (rescaled to mean 0, std 1) before either
method runs, since the raw features live on wildly different scales
(avg_amount is in the millions of TWD; near_threshold_ratio is 0-1;
max_countries_in_7d is a small integer). Without standardizing,
large-magnitude features would dominate distance calculations
regardless of how informative they actually are.
"""

from __future__ import annotations

import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "n_transactions", "total_sent", "total_received", "avg_amount",
    "std_amount", "max_amount", "unique_counterparties",
    "max_countries_in_7d", "txn_per_day", "max_txns_in_24h",
    "near_threshold_count", "near_threshold_ratio", "in_out_ratio",
    "avg_hold_time_hours", "va_code_share",
    "pct_19d_professional_services", "pct_masked_purpose",
]


def score_anomalies(
    feat_df: pd.DataFrame,
    contamination: float = 0.08,
    dbscan_eps: float = 1.2,
    dbscan_min_samples: int = 4,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Adds anomaly_score, is_isoforest_outlier, cluster_label, and
    is_density_outlier columns to a copy of feat_df.

    contamination: our rough prior on what fraction of entities are
    actually anomalous -- 0.08 means "assume ~8% of entities are
    unusual." This doesn't need to be exact; it just anchors where
    Isolation Forest draws its outlier/normal cutoff.
    """
    out = feat_df.copy()
    X = out[FEATURE_COLUMNS].values
    X_scaled = StandardScaler().fit_transform(X)

    iso = IsolationForest(
        contamination=contamination, random_state=random_state, n_estimators=300
    )
    iso.fit(X_scaled)
    # decision_function: higher = more normal by sklearn's convention.
    # We flip the sign so higher = more anomalous, which reads more
    # naturally downstream (a "risk score" should go up with risk).
    out["anomaly_score"] = -iso.decision_function(X_scaled)
    out["is_isoforest_outlier"] = iso.predict(X_scaled) == -1

    db = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
    cluster_labels = db.fit_predict(X_scaled)
    out["cluster_label"] = cluster_labels
    out["is_density_outlier"] = cluster_labels == -1

    # Rescale the raw anomaly_score to a readable 0-100 range -- the
    # raw sklearn score is technically meaningful but not intuitive to
    # look at directly.
    lo, hi = out["anomaly_score"].min(), out["anomaly_score"].max()
    out["anomaly_score_0_100"] = (
        (out["anomaly_score"] - lo) / (hi - lo) * 100 if hi > lo else 0.0
    )

    return out.sort_values("anomaly_score_0_100", ascending=False)


if __name__ == "__main__":
    from aml_pipeline.data_generator import GeneratorConfig, generate_transactions
    from aml_pipeline.features import build_entity_features

    txns, entities = generate_transactions(GeneratorConfig())
    feats = build_entity_features(txns, entities)
    scored = score_anomalies(feats)

    print("Top 10 entities by anomaly score:")
    print(scored[[
        "entity_id", "segment", "anomaly_score_0_100",
        "is_isoforest_outlier", "is_density_outlier", "true_pattern"
    ]].head(10).to_string())

    print("\nHow many injected-pattern entities land in the top 20% by score?")
    n = len(scored)
    top_20pct = scored.head(int(n * 0.2))
    n_injected_total = (scored["true_pattern"] != "NONE").sum()
    n_injected_in_top = (top_20pct["true_pattern"] != "NONE").sum()
    print(f"{n_injected_in_top} of {n_injected_total} injected entities in top 20% ({int(n*0.2)} entities)")
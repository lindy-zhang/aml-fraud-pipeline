# Cross-Border Remittance Transaction Monitoring

A transaction-monitoring pipeline built over **synthetic** cross-border remittance data, modeled on the shui-dan (水單) / remittance-advice record structure used in correspondent banking compliance workflows. Built to explore how unsupervised anomaly detection and network analysis can surface money-laundering typologies without relying on scarce, late-arriving fraud labels.

> **Note:** All data here is synthetically generated. No real client, account, or transaction data is used anywhere in this project.

## Status: in progress 🚧

This is being built incrementally, stage by stage, with each stage tested before moving to the next. Current state:

- [x] Synthetic transaction + entity generator (`src/aml_pipeline/data_generator.py`)
- [ ] Injected fraud typologies (structuring, layering, fan-out/fan-in)
- [ ] Entity-level feature engineering
- [ ] Unsupervised anomaly detection (Isolation Forest + DBSCAN)
- [ ] Counterparty network / graph analysis
- [ ] Rule engine converting findings into an auditable alert queue
- [ ] Notebook walkthrough
- [ ] Streamlit dashboard

## Why this project

Confirmed money-laundering labels are rare, arrive late (often months after a transaction, if at all), and the typologies that produce them evolve specifically to evade whatever rule caught the last case. That makes this a poor fit for supervised classification and a better fit for **unsupervised structure-finding**: clustering and outlier detection over behavioral features, combined with network analysis over the counterparty graph, with findings converted into explicit rules a human analyst can review.

## Getting started

```bash
git clone https://github.com/lindy-zhang/aml-fraud-pipeline.git
cd aml-fraud-pipeline

python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Run the data generator on its own:

```bash
python3 src/aml_pipeline/data_generator.py
```

## Repo structure
# H3 Repository

This repository contains the H3 link prediction method, structural baselines, and GNN/embedding baselines.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Fast demo:

```bash
python run_demo.py
```

- Default variant: `norm_default`
- Default negative ratio: `20x`
- Demo now uses the full positive edge set by default

## H3 Only

Formal H3 run without demo:

```bash
python run_h3.py
```

This uses:

- mode: `full`
- variant: `norm_default`
- negative ratio: `20`
- data: `data/2014AK_30.csv` and `data/2014AK_90.csv`

Useful variants:

```bash
python run_h3.py --task A
python run_h3.py --variant norm_default
python run_h3.py --variant path_linear
python run_h3.py --negative-ratio 20
```

Variant sweep from the H3 core config:

```bash
python h3/h3_core.py --config h3/h3_config.json
python h3/h3_core.py --config h3/h3_config.json --variants norm_default
```

## Structural Baselines

Task A:

```bash
python structural_methods/run_non_gnn.py --task A --variants norm_default
```

Task B:

```bash
python structural_methods/run_non_gnn.py --task B --cross-windows 90 --variants norm_default
```

Notes:

- default negative ratio is `20x`
- `--variants norm_default` keeps H3 on the default config while comparing against other heuristics

## Learning Methods

Task A:

```bash
python learning_methods/run_learning_methods.py --task A --models GCN GraphSAGE GAT
```

Task B:

```bash
python learning_methods/run_learning_methods.py --task B --cross-windows 90 --models GCN GraphSAGE GAT
```

## Data

The current layout expects demo data directly under `data/`:

```text
data/
|-- 2014AK_30.csv
|-- 2014AK_90.csv
`-- 2014AK_180.csv
```

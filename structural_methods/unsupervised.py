"""
Unsupervised evaluation harness for H3 and classical heuristics.

References:
- H3 scoring logic and variants: h3_core.H3Variant / h3_score
- Baseline heuristics: common neighbors, Adamic-Adar, Preferential Attachment, L3
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
from tqdm import tqdm

from utils.metrics import build_recall_grid, compute_metrics, aggregate_metrics

from h3.h3_core import (
    GraphCache,
    build_candidate_pairs,
    sample_candidate_pairs,
    prepare_dataset,
    resolve_h3_datasets,
    set_all_seeds,
    split_train_test,
)


# -------------------- Baseline heuristics -------------------- #



def l3_kovacs_score(G, x, y, cache: GraphCache | None = None) -> float:
    """
    L3 score following the Kovacs reference implementation (suggestions.txt).
    Uses degree = sum of absolute edge weights, and accumulates
    sum_{A,B in paths x-A-B-y} 1/sqrt(deg(A)*deg(B)).
    """
    if cache is None:
        if x not in G or y not in G:
            return 0.0
        neighbors = lambda n: G.neighbors(n)
        degree = lambda n: sum(abs(G[n][nbr].get("weight", 1.0)) for nbr in G.neighbors(n))
    else:
        if x not in cache.node_to_idx or y not in cache.node_to_idx:
            return 0.0
        neighbors = cache.neighbors
        degree = lambda n: sum(abs(cache.edge_weight(n, nbr)) for nbr in cache.neighbors(n))

    score = 0.0
    for a in neighbors(x):
        deg_a = degree(a)
        if deg_a <= 0:
            continue
        for b in neighbors(a):
            if b == x:
                continue
            if y not in set(neighbors(b)):
                continue
            deg_b = degree(b)
            if deg_b <= 0:
                continue
            score += 1.0 / math.sqrt(deg_a * deg_b)
    return score

def common_neighbor_score(G, x, y, cache: GraphCache | None = None) -> float:
    if cache is None:
        if x not in G or y not in G:
            return 0.0
        return float(len(set(nx.common_neighbors(G, x, y))))
    if x not in cache.node_to_idx or y not in cache.node_to_idx:
        return 0.0
    return float(len(cache.neighbors(x) & cache.neighbors(y)))


def adamic_adar_score(G, x, y, cache: GraphCache | None = None) -> float:
    if cache is None:
        if x not in G or y not in G:
            return 0.0
        shared = set(nx.common_neighbors(G, x, y))
        if not shared:
            return 0.0
        return sum(1.0 / np.log(max(G.degree(z), 2)) for z in shared)
    if x not in cache.node_to_idx or y not in cache.node_to_idx:
        return 0.0
    shared = cache.neighbors(x) & cache.neighbors(y)
    if not shared:
        return 0.0
    degs = cache.degree_array[[cache.node_to_idx[node] for node in shared]]
    adjusted = np.where(degs > 1.0, degs, 2.0)
    return float(np.sum(1.0 / np.log(adjusted)))


def preferential_attachment_score(G, x, y, cache: GraphCache | None = None) -> float:
    if cache is None:
        if x not in G or y not in G:
            return 0.0
        return float(G.degree(x) * G.degree(y))
    return float(cache.degree(x) * cache.degree(y))


def hub_promoted_score(G, x, y, cache: GraphCache | None = None) -> float:
    if cache is None:
        if x not in G or y not in G:
            return 0.0
        deg_x = G.degree(x)
        deg_y = G.degree(y)
        denom = min(deg_x, deg_y)
        if denom <= 0:
            return 0.0
        shared = set(nx.common_neighbors(G, x, y))
    else:
        if x not in cache.node_to_idx or y not in cache.node_to_idx:
            return 0.0
        deg_x = cache.degree(x)
        deg_y = cache.degree(y)
        denom = min(deg_x, deg_y)
        if denom <= 0:
            return 0.0
        shared = cache.neighbors(x) & cache.neighbors(y)
    if not shared:
        return 0.0
    return float(len(shared) / denom)


def leicht_holme_newman_score(G, x, y, cache: GraphCache | None = None) -> float:
    if cache is None:
        if x not in G or y not in G:
            return 0.0
        deg_x = G.degree(x)
        deg_y = G.degree(y)
        denom = deg_x * deg_y
        if denom <= 0:
            return 0.0
        shared = set(nx.common_neighbors(G, x, y))
    else:
        if x not in cache.node_to_idx or y not in cache.node_to_idx:
            return 0.0
        deg_x = cache.degree(x)
        deg_y = cache.degree(y)
        denom = deg_x * deg_y
        if denom <= 0:
            return 0.0
        shared = cache.neighbors(x) & cache.neighbors(y)
    if not shared:
        return 0.0
    return float(len(shared) / denom)


def jaccard_score(G, x, y, cache: GraphCache | None = None) -> float:
    if cache is None:
        if x not in G or y not in G:
            return 0.0
        neigh_x = set(G.neighbors(x))
        neigh_y = set(G.neighbors(y))
    else:
        if x not in cache.node_to_idx or y not in cache.node_to_idx:
            return 0.0
        neigh_x = cache.neighbors(x)
        neigh_y = cache.neighbors(y)
    union_size = len(neigh_x | neigh_y)
    if union_size == 0:
        return 0.0
    inter_size = len(neigh_x & neigh_y)
    return float(inter_size / union_size)


def resource_allocation_score(G, x, y, cache: GraphCache | None = None) -> float:
    if cache is None:
        if x not in G or y not in G:
            return 0.0
        shared = set(nx.common_neighbors(G, x, y))
        if not shared:
            return 0.0
        total = 0.0
        for z in shared:
            deg_z = G.degree(z)
            if deg_z > 0:
                total += 1.0 / deg_z
        return float(total)
    if x not in cache.node_to_idx or y not in cache.node_to_idx:
        return 0.0
    shared = cache.neighbors(x) & cache.neighbors(y)
    if not shared:
        return 0.0
    total = 0.0
    for z in shared:
        deg_z = cache.degree(z)
        if deg_z > 0:
            total += 1.0 / deg_z
    return float(total)


# -------------------- Core runner -------------------- #




def _score_method(
    method: str,
    G_train: nx.Graph,
    cache: GraphCache,
    candidate_pairs: Sequence[Tuple],
):
    scores = []
    if method == "l3":
        for u, v in candidate_pairs:
            scores.append(l3_kovacs_score(G_train, u, v, cache=cache))
    elif method == "aa":
        for u, v in candidate_pairs:
            scores.append(adamic_adar_score(G_train, u, v, cache=cache))
    elif method == "cn":
        for u, v in candidate_pairs:
            scores.append(common_neighbor_score(G_train, u, v, cache=cache))
    elif method == "pa":
        for u, v in candidate_pairs:
            scores.append(preferential_attachment_score(G_train, u, v, cache=cache))
    elif method in ("hp", "hpi", "hub_promoted"):
        for u, v in candidate_pairs:
            scores.append(hub_promoted_score(G_train, u, v, cache=cache))
    elif method in ("lhn", "lhn1", "leicht_holme_newman"):
        for u, v in candidate_pairs:
            scores.append(leicht_holme_newman_score(G_train, u, v, cache=cache))
    elif method == "jaccard":
        for u, v in candidate_pairs:
            scores.append(jaccard_score(G_train, u, v, cache=cache))
    elif method in ("ra", "resource_allocation"):
        for u, v in candidate_pairs:
            scores.append(resource_allocation_score(G_train, u, v, cache=cache))
    else:
        raise ValueError(f"Unknown method '{method}'")
    return scores


def run_unsupervised(
    config_path: Path = Path("h3_config.json"),
    dataset_override: Sequence[Path] | None = None,
    skip_existing: bool = False,
) -> Dict:
    cfg_path = config_path.resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    dataset_entry = cfg["dataset"]
    dataset_pattern = cfg.get("dataset_pattern", "*_30.csv")
    weight_column_cfg = cfg.get("weight_column")
    if isinstance(weight_column_cfg, (list, tuple)):
        weight_column = weight_column_cfg[0]
    else:
        weight_column = weight_column_cfg
    trials = int(cfg.get("trials", 5))
    test_size = float(cfg.get("test_size", 0.5))
    recall_step = float(cfg.get("recall_step", 0.01))
    recall_end = float(cfg.get("recall_end", 0.1))
    base_seed = int(cfg.get("base_seed", 42))
    max_candidates_cfg = cfg.get("max_candidate_pairs", "auto")
    auto_factor = float(cfg.get("auto_candidate_factor", 3.0))
    auto_ceiling = cfg.get("auto_candidate_ceiling", 10_000_000)
    bucket_quantiles = cfg.get("candidate_bucket_quantiles", [50.0, 90.0])
    bucket_weights = cfg.get("candidate_bucket_weights", [0.3, 0.5, 0.2])
    min_per_node = int(cfg.get("candidate_min_per_node", 0))

    base_methods = cfg.get("unsupervised", {}).get("methods", ["l3", "aa", "cn", "pa", "hp", "lhn", "jaccard", "ra"])
    recall_grid = build_recall_grid(step=recall_step, end=recall_end)

    dataset_paths = list(dataset_override) if dataset_override else resolve_h3_datasets(dataset_entry, pattern=dataset_pattern)

    # Map dataset path -> output path; if dataset_override was provided or multiple datasets exist,
    # always split outputs per dataset to avoid clobbering across runs.
    output_cfg = cfg.get("unsupervised", {}).get("output", "reports/unsupervised.json")
    base_output = Path(output_cfg)
    force_split = bool(dataset_override) or len(dataset_paths) > 1
    if base_output.suffix:
        output_dir = base_output.parent
        if force_split:
            dataset_to_output = {
                p: output_dir / f"{p.stem}_{base_output.stem}{base_output.suffix}" for p in dataset_paths
            }
        else:
            dataset_to_output = {dataset_paths[0]: base_output}
    else:
        output_dir = base_output
        dataset_to_output = {p: output_dir / f"{p.stem}_unsupervised.json" for p in dataset_paths}

    combined_outputs = {}
    for dataset_path in dataset_paths:
        if not dataset_path.is_file():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        output_path = dataset_to_output[dataset_path]
        if skip_existing:
            existing_payload = None
            existing_path = None

            def _matches_dataset(value) -> bool:
                if not value:
                    return False
                try:
                    return Path(value).name == dataset_path.name
                except Exception:
                    return False

            if output_path.is_file():
                existing_path = output_path
                try:
                    existing_payload = json.loads(output_path.read_text(encoding="utf-8-sig"))
                except Exception as exc:  # pragma: no cover - defensive fallback
                    print(f"[Warn] Failed to read existing file, recomputing: {exc}")
            else:
                # Legacy combined file support
                base_output = Path(output_cfg)
                if base_output.is_file():
                    try:
                        base_json = json.loads(base_output.read_text(encoding="utf-8-sig"))
                        if isinstance(base_json, dict):
                            if _matches_dataset(base_json.get("dataset")) or base_json.get("dataset") == str(dataset_path):
                                existing_payload = base_json
                                existing_path = base_output
                            elif str(dataset_path) in base_json:
                                existing_payload = base_json[str(dataset_path)]
                                existing_path = base_output
                            else:
                                for key, val in base_json.items():
                                    if _matches_dataset(key):
                                        existing_payload = val
                                        existing_path = base_output
                                        break
                    except Exception as exc:  # pragma: no cover - defensive fallback
                        print(f"[Warn] Failed to read combined unsupervised file, recomputing: {exc}")
            if existing_payload is not None:
                print(f"[Skip] Unsupervised existing results found at {existing_path}")
                combined_outputs[str(dataset_path)] = existing_payload
                continue

        print(f"[Start] Unsupervised | {dataset_path.name}")
        data = prepare_dataset(dataset_path, weight_col=weight_column)
        total_nodes = sorted(set(data["node1"]).union(set(data["node2"])))

        all_results = {}
        method_list = []
        for method in base_methods:
            if method.lower() == "h3":
                # Skip H3 per request to keep only non-H3 baselines
                continue
            method_list.append((method, method, None))

        outer = tqdm(range(trials), desc=f"{dataset_path.name} trials", unit="t")
        for trial_idx in outer:
            seed = base_seed + trial_idx
            set_all_seeds(seed)
            X_train, X_test = split_train_test(data, test_size=test_size, seed=seed)

            G_train = nx.Graph()
            for _, row in X_train.iterrows():
                G_train.add_edge(row["node1"], row["node2"], weight=row["weight"])
            G_train.add_nodes_from(total_nodes)

            test_edges = [tuple(edge) for edge in X_test[["node1", "node2"]].itertuples(index=False, name=None)]
            test_edge_set = {tuple(sorted(edge)) for edge in test_edges}
            candidate_pairs = build_candidate_pairs(G_train, total_nodes, test_edges)
            candidate_pairs = sample_candidate_pairs(
                G_train,
                candidate_pairs,
                test_edges,
                cap_value=max_candidates_cfg,
                seed=seed,
                auto_factor=auto_factor,
                auto_ceiling=auto_ceiling,
                quantiles=bucket_quantiles,
                bucket_weights=bucket_weights,
                min_per_node=min_per_node,
            )
            cache = GraphCache.from_graph(G_train, total_nodes)

            inner = tqdm(method_list, desc="methods", leave=False)
            for display_name, method_key, _ in inner:
                scores = _score_method(method_key, G_train, cache, candidate_pairs)
                metrics = compute_metrics(
                    candidate_pairs,
                    scores,
                    test_edge_set,
                    recall_grid,
                    total_nodes=total_nodes,
                )

                bucket = all_results.setdefault(display_name, {"metrics": []})
                bucket["metrics"].append(metrics)
                inner.set_postfix({display_name: f"{metrics['precision_at_recall_mean']:.4f}"})
            inner.close()
        outer.close()

        summary = {}
        for name, payload in all_results.items():
            stats = aggregate_metrics(payload["metrics"])
            summary[name] = stats

        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_payload = {
            "dataset": str(dataset_path),
            "trials": trials,
            "test_size": test_size,
            "recall_step": recall_step,
            "recall_end": recall_end,
            "recall_grid": recall_grid,
            "methods": summary,
        }
        output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")

        print(f"[Done] Unsupervised results saved to {output_path}")
        for name, stats in summary.items():
            print(f"  {name}: precision@recall_mean={stats['precision_at_recall_mean']:.6f}")

        combined_outputs[str(dataset_path)] = output_payload

    if len(combined_outputs) == 1:
        return next(iter(combined_outputs.values()))
    return combined_outputs


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run unsupervised baseline evaluations.")
    parser.add_argument("--config", type=Path, default=Path("h3_config.json"), help="Path to config JSON.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip datasets whose outputs already exist.",
    )
    args = parser.parse_args()

    run_unsupervised(args.config, skip_existing=args.resume)

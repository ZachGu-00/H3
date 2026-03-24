#!/usr/bin/env python3
"""
Shared H3 demo/evaluation runner.

- `run_demo.py` uses quick mode by default.
- `run_h3.py` uses full mode by default.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import networkx as nx

sys.path.insert(0, str(Path(__file__).parent.parent))

from h3.h3_core import GraphCache, h3_score, load_h3_variants, prepare_dataset, set_all_seeds, split_train_test
from utils.metrics import build_recall_grid, compute_metrics


ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
DEFAULT_DATA_30 = DATA_DIR / "2014AK_30.csv"
DEFAULT_DATA_90 = DATA_DIR / "2014AK_90.csv"
DEFAULT_CONFIG = ROOT_DIR / "h3" / "h3_config.json"
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "demo_results.json"


def load_variant_params(config_path: Path, variant_name: str) -> Dict[str, float]:
    variants = load_h3_variants(config_path)
    if not variants:
        raise ValueError(f"No H3 variants found in {config_path}")
    for variant in variants:
        if variant.name == variant_name:
            return {
                "name": variant.name,
                "forward_weight": variant.forward_weight,
                "reverse_weight": variant.reverse_weight,
                "penalty_gamma": variant.penalty_gamma,
                "min_penalty": variant.min_penalty,
                "connector_gamma": variant.connector_gamma,
                "target_gamma": variant.target_gamma,
                "path_weight_gamma": variant.path_weight_gamma,
            }
    available = ", ".join(v.name for v in variants)
    raise ValueError(f"Unknown H3 variant '{variant_name}'. Available: {available}")


def sample_distance2_negatives(
    graph: nx.Graph,
    nodes: Sequence,
    banned_edge_set: set[Tuple],
    desired_negatives: int,
    seed: int,
) -> List[Tuple]:
    if desired_negatives <= 0 or len(nodes) < 2:
        return []

    node_set = set(nodes)
    candidates: set[Tuple] = set()
    for u in node_set:
        if u not in graph:
            continue
        neighbors_u = set(graph.neighbors(u))
        for mid in neighbors_u:
            for v in graph.neighbors(mid):
                if v == u or v not in node_set or v in neighbors_u:
                    continue
                pair = (u, v) if u < v else (v, u)
                if pair not in banned_edge_set:
                    candidates.add(pair)

    candidates_list = list(candidates)
    random.Random(seed).shuffle(candidates_list)
    return sorted(candidates_list[:desired_negatives])


def maybe_limit_edges(edges: Sequence[Tuple], max_edges: int | None, seed: int) -> List[Tuple]:
    edges_list = list(edges)
    if not max_edges or max_edges <= 0 or len(edges_list) <= max_edges:
        return sorted(edges_list)
    rng = random.Random(seed)
    rng.shuffle(edges_list)
    return sorted(edges_list[:max_edges])


def score_pairs(pairs: Sequence[Tuple], cache: GraphCache, variant_params: Dict[str, float]) -> List[float]:
    return [
        float(
            h3_score(
                None,
                u,
                v,
                cache=cache,
                forward_weight=variant_params["forward_weight"],
                reverse_weight=variant_params["reverse_weight"],
                penalty_gamma=variant_params["penalty_gamma"],
                min_penalty=variant_params["min_penalty"],
                connector_gamma=variant_params["connector_gamma"],
                target_gamma=variant_params["target_gamma"],
                path_weight_gamma=variant_params["path_weight_gamma"],
            )
        )
        for u, v in pairs
    ]


def build_graph(rows, total_nodes: Iterable) -> nx.Graph:
    graph = nx.Graph()
    for _, row in rows.iterrows():
        graph.add_edge(row["node1"], row["node2"], weight=float(row["weight"]))
    graph.add_nodes_from(total_nodes)
    return graph


def extract_metrics(metrics: Dict) -> Dict[str, float]:
    return {
        "AUROC": metrics["auroc"],
        "AUPRC": metrics["auprc"],
        "P@Recall": metrics["precision_at_recall_mean"],
        "MRR": metrics["mrr"],
        "MRP": metrics["mean_rank_percentile"],
    }


def run_task_a(
    data_30_path: Path,
    recall_grid: Sequence[float],
    negative_ratio: int,
    max_positive_edges: int | None,
    seed: int,
    variant_params: Dict[str, float],
) -> Dict[str, float]:
    df = prepare_dataset(data_30_path, weight_col="benecount")
    total_nodes = sorted(set(df["node1"]).union(df["node2"]))
    train_rows, test_rows = split_train_test(df, test_size=0.5, seed=seed)

    graph = build_graph(train_rows, total_nodes)
    all_edges = {
        tuple(sorted(edge))
        for edge in df[["node1", "node2"]].itertuples(index=False, name=None)
    }
    positive_edges = {
        tuple(sorted(edge))
        for edge in test_rows[["node1", "node2"]].itertuples(index=False, name=None)
    }
    positive_edges = maybe_limit_edges(sorted(positive_edges), max_positive_edges, seed)
    test_nodes = sorted({node for edge in positive_edges for node in edge})
    negative_edges = sample_distance2_negatives(
        graph,
        test_nodes,
        all_edges,
        max(1, len(positive_edges) * negative_ratio),
        seed + 1,
    )

    candidate_pairs = positive_edges + negative_edges
    cache = GraphCache.from_graph(graph, total_nodes)
    scores = score_pairs(candidate_pairs, cache, variant_params)
    metrics = compute_metrics(candidate_pairs, scores, set(positive_edges), recall_grid)
    summary = extract_metrics(metrics)
    summary["positives"] = len(positive_edges)
    summary["negatives"] = len(negative_edges)
    return summary


def run_task_b(
    data_30_path: Path,
    data_90_path: Path,
    recall_grid: Sequence[float],
    negative_ratio: int,
    max_positive_edges: int | None,
    seed: int,
    variant_params: Dict[str, float],
) -> Dict[str, float]:
    df_30 = prepare_dataset(data_30_path, weight_col="benecount")
    df_90 = prepare_dataset(data_90_path, weight_col="benecount")
    total_nodes = sorted(set(df_30["node1"]).union(df_30["node2"]).union(df_90["node1"]).union(df_90["node2"]))

    graph = build_graph(df_30, total_nodes)
    train_edges = {
        tuple(sorted(edge))
        for edge in df_30[["node1", "node2"]].itertuples(index=False, name=None)
    }
    test_edges_all = {
        tuple(sorted(edge))
        for edge in df_90[["node1", "node2"]].itertuples(index=False, name=None)
    }
    positive_edges = maybe_limit_edges(sorted(test_edges_all - train_edges), max_positive_edges, seed + 10)
    test_nodes = sorted({node for edge in positive_edges for node in edge})
    negative_edges = sample_distance2_negatives(
        graph,
        test_nodes,
        test_edges_all,
        max(1, len(positive_edges) * negative_ratio),
        seed + 2,
    )

    candidate_pairs = positive_edges + negative_edges
    cache = GraphCache.from_graph(graph, total_nodes)
    scores = score_pairs(candidate_pairs, cache, variant_params)
    metrics = compute_metrics(candidate_pairs, scores, set(positive_edges), recall_grid)
    summary = extract_metrics(metrics)
    summary["positives"] = len(positive_edges)
    summary["negatives"] = len(negative_edges)
    return summary


def print_matrix(results: Dict[str, Dict[str, float]]) -> None:
    headers = ["Task", "AUROC", "AUPRC", "P@Recall", "MRR", "MRP", "Pos", "Neg"]
    rows = []
    for task_name, values in results.items():
        rows.append(
            [
                task_name,
                f"{values['AUROC']:.4f}",
                f"{values['AUPRC']:.4f}",
                f"{values['P@Recall']:.4f}",
                f"{values['MRR']:.4f}",
                f"{values['MRP']:.4f}",
                str(values["positives"]),
                str(values["negatives"]),
            ]
        )

    widths = [max(len(headers[idx]), max(len(row[idx]) for row in rows)) for idx in range(len(headers))]
    print(" | ".join(headers[idx].ljust(widths[idx]) for idx in range(len(headers))))
    print("-+-".join("-" * widths[idx] for idx in range(len(headers))))
    for row in rows:
        print(" | ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))))


def parse_args(argv: Sequence[str] | None = None, default_mode: str = "quick") -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run H3 and print an evaluation matrix.")
    parser.add_argument("--task", choices=["A", "B", "both"], default="both")
    parser.add_argument("--mode", choices=["quick", "full"], default=default_mode)
    parser.add_argument("--data-30", type=Path, default=DEFAULT_DATA_30)
    parser.add_argument("--data-90", type=Path, default=DEFAULT_DATA_90)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--variant", type=str, default="norm_default")
    parser.add_argument("--negative-ratio", type=int, default=20)
    parser.add_argument("--max-positive-edges", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None, default_mode: str = "quick") -> None:
    args = parse_args(argv=argv, default_mode=default_mode)
    set_all_seeds(args.seed)

    if not args.data_30.is_file():
        raise FileNotFoundError(f"Missing dataset: {args.data_30}")
    if args.task in {"B", "both"} and not args.data_90.is_file():
        raise FileNotFoundError(f"Missing dataset: {args.data_90}")

    quick_mode = args.mode == "quick"
    max_positive_edges = args.max_positive_edges
    recall_grid = build_recall_grid(step=0.01, end=0.1)
    variant_params = load_variant_params(args.config, args.variant)

    results: Dict[str, Dict[str, float]] = {}
    if args.task in {"A", "both"}:
        results["Task A"] = run_task_a(
            args.data_30,
            recall_grid,
            negative_ratio=args.negative_ratio,
            max_positive_edges=max_positive_edges,
            seed=args.seed,
            variant_params=variant_params,
        )
    if args.task in {"B", "both"}:
        results["Task B"] = run_task_b(
            args.data_30,
            args.data_90,
            recall_grid,
            negative_ratio=args.negative_ratio,
            max_positive_edges=max_positive_edges,
            seed=args.seed,
            variant_params=variant_params,
        )

    print(f"H3 mode: {args.mode}")
    print(f"H3 variant: {variant_params['name']}")
    print(f"Negative ratio: {args.negative_ratio}x")
    print(f"Data 30: {args.data_30.name}")
    if args.task in {"B", "both"}:
        print(f"Data 90: {args.data_90.name}")
    print_matrix(results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "mode": args.mode,
                "task": args.task,
                "seed": args.seed,
                "config": str(args.config),
                "variant": variant_params["name"],
                "negative_ratio": args.negative_ratio,
                "max_positive_edges": max_positive_edges,
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nSaved results to: {args.output}")


if __name__ == "__main__":
    main()

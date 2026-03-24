from __future__ import annotations

import argparse
import concurrent.futures
import json
import math
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
import sys
from pathlib import Path as PathLib

from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(PathLib(__file__).parent.parent))

from h3.h3_core import (
    GraphCache,
    h3_score,
    load_h3_variants,
    prepare_dataset,
    set_all_seeds,
    split_train_test,
)

from utils.metrics import (
    aggregate_metrics,
    build_recall_grid,
    compute_metrics,
    ranking_metrics_by_source,
)
from structural_methods.unsupervised import (
    adamic_adar_score,
    common_neighbor_score,
    hub_promoted_score,
    jaccard_score,
    leicht_holme_newman_score,
    preferential_attachment_score,
    resource_allocation_score,
    l3_kovacs_score,
)


DATASET_ROOTS = [
    Path("data/2014data"),
    Path("data/2015data"),
]
DATASET_PATTERN = "*_30.csv"
MAX_EDGE_ROWS = 10_000_000
TASK_WITHIN = "A"
TASK_CROSS_PERIOD = "B"


def _count_edges(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8") as f:
        next(f, None)
        return sum(1 for _ in f)


def _resolve_dataset_paths(max_edge_rows: int) -> List[Path]:
    paths: List[Path] = []
    for root in DATASET_ROOTS:
        if not root.is_dir():
            continue
        for path in sorted(root.glob(DATASET_PATTERN)):
            if not path.is_file():
                continue
            edge_count = _count_edges(path)
            if edge_count > max_edge_rows:
                print(f"[Skip] {path} (edges={edge_count} > {max_edge_rows})")
                continue
            paths.append(path)
    return paths


def _parse_dataset_info(path: Path) -> Tuple[str, str, int] | None:
    match = re.match(r"^(?P<year>\d{4})(?P<state>[A-Z]{2})_(?P<window>\d+)$", path.stem)
    if not match:
        return None
    year = match.group("year")
    state = match.group("state")
    window = int(match.group("window"))
    return year, state, window


def _index_datasets() -> Dict[Tuple[str, str, int], Path]:
    index: Dict[Tuple[str, str, int], Path] = {}
    for root in DATASET_ROOTS:
        if not root.is_dir():
            continue
        for path in sorted(root.glob("*.csv")):
            if not path.is_file():
                continue
            info = _parse_dataset_info(path)
            if info:
                index[info] = path
    return index


def _resolve_task_b_pairs(
    max_edge_rows: int,
    windows: Sequence[int],
    limit_long_edges: bool = True,
) -> List[Tuple[Path, Path, int]]:
    index = _index_datasets()
    pairs: List[Tuple[Path, Path, int]] = []
    for (year, state, window), train_path in index.items():
        if window != 30:
            continue
        train_edges = _count_edges(train_path)
        if train_edges > max_edge_rows:
            print(f"[Skip] {train_path} (edges={train_edges} > {max_edge_rows})")
            continue
        for long_window in windows:
            test_path = index.get((year, state, int(long_window)))
            if not test_path:
                continue
            if limit_long_edges:
                test_edges = _count_edges(test_path)
                if test_edges > max_edge_rows:
                    print(f"[Skip] {test_path} (edges={test_edges} > {max_edge_rows})")
                    continue
            pairs.append((train_path, test_path, int(long_window)))
    return pairs


def _edge_set_from_df(df) -> List[Tuple]:
    edges = set()
    for u, v in df[["node1", "node2"]].itertuples(index=False, name=None):
        if u == v:
            continue
        pair = (u, v) if u < v else (v, u)
        edges.add(pair)
    return sorted(edges)


def _sample_negative_pairs(
    nodes: Sequence,
    banned_edge_set: set[Tuple],
    desired_negatives: int,
    seed: int,
    max_trials_factor: int = 25,
) -> List[Tuple]:
    if desired_negatives <= 0 or len(nodes) < 2:
        return []
    rng = random.Random(seed)
    nodes_list = list(nodes)
    negatives: set[Tuple] = set()
    max_trials = max_trials_factor * desired_negatives
    trials = 0
    while len(negatives) < desired_negatives and trials < max_trials:
        u = rng.choice(nodes_list)
        v = rng.choice(nodes_list)
        trials += 1
        if u == v:
            continue
        pair = (u, v) if u < v else (v, u)
        if pair in banned_edge_set or pair in negatives:
            continue
        negatives.add(pair)
    return sorted(negatives)


def _sample_distance2_negatives(
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
        for w in neighbors_u:
            for v in graph.neighbors(w):
                if v == u or v not in node_set:
                    continue
                if v in neighbors_u:
                    continue
                pair = (u, v) if u < v else (v, u)
                if pair in banned_edge_set:
                    continue
                candidates.add(pair)
    if not candidates:
        return []
    candidates_list = list(candidates)
    rng = random.Random(seed)
    rng.shuffle(candidates_list)
    if len(candidates_list) > desired_negatives:
        candidates_list = candidates_list[:desired_negatives]
    return sorted(candidates_list)


_WORKER_STATE = {"cache": None, "method": None, "params": None, "deg_map": None}





def _baseline_worker_init(cache: GraphCache, method: str) -> None:
    _WORKER_STATE["cache"] = cache
    _WORKER_STATE["method"] = method
    _WORKER_STATE["params"] = None
    if method.lower() == "l3":
        deg_map = {}
        for node in cache.node_to_idx.keys():
            total = 0.0
            for nbr in cache.neighbors(node):
                total += abs(cache.edge_weight(node, nbr))
            deg_map[node] = total
        _WORKER_STATE["deg_map"] = deg_map
    else:
        _WORKER_STATE["deg_map"] = None


def _baseline_score_worker(pair) -> float:
    cache = _WORKER_STATE["cache"]
    method = _WORKER_STATE["method"]
    deg_map = _WORKER_STATE.get("deg_map")
    u, v = pair
    return _score_baseline(method, u, v, cache, deg_map)


def _h3_worker_init(cache: GraphCache, params: Dict) -> None:
    _WORKER_STATE["cache"] = cache
    _WORKER_STATE["method"] = None
    _WORKER_STATE["params"] = params


def _h3_score_worker(pair) -> float:
    cache = _WORKER_STATE["cache"]
    params = _WORKER_STATE["params"]
    u, v = pair
    return h3_score(
        None,
        u,
        v,
        cache=cache,
        forward_weight=params["forward_weight"],
        reverse_weight=params["reverse_weight"],
        penalty_gamma=params["penalty_gamma"],
        min_penalty=params["min_penalty"],
        connector_gamma=params["connector_gamma"],
        target_gamma=params["target_gamma"],
        path_weight_gamma=params["path_weight_gamma"],
    )


def _score_pairs_parallel(
    pairs,
    max_workers,
    init_fn,
    init_args,
    worker_fn,
    chunksize=128,
    desc: str | None = None,
    show_progress: bool = False,
):
    if max_workers <= 1 or not pairs:
        init_fn(*init_args)
        return [worker_fn(pair) for pair in pairs]
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_fn,
        initargs=init_args,
    ) as executor:
        if not show_progress:
            return list(executor.map(worker_fn, pairs, chunksize=chunksize))
        results = []
        total = len(pairs)
        bar = tqdm(total=total, desc=desc, leave=False)
        for s in executor.map(worker_fn, pairs, chunksize=chunksize):
            results.append(s)
            bar.update(1)
        bar.close()
        return results


def _l3_kovacs_score_cached(u, v, cache: GraphCache, deg_map: Dict | None) -> float:
    if u not in cache.node_to_idx or v not in cache.node_to_idx:
        return 0.0
    if deg_map is None:
        return float(l3_kovacs_score(None, u, v, cache=cache))
    score = 0.0
    for a in cache.neighbors(u):
        deg_a = deg_map.get(a, 0.0)
        if deg_a <= 0:
            continue
        for b in cache.neighbors(a):
            if cache.has_edge(b, v):
                deg_b = deg_map.get(b, 0.0)
                if deg_b > 0:
                    score += 1.0 / math.sqrt(deg_a * deg_b)
    return score


def _score_baseline(method: str, u, v, cache: GraphCache, deg_map: Dict | None = None) -> float:
    m = method.lower()
    if m == "l3":
        return float(_l3_kovacs_score_cached(u, v, cache, deg_map))
    if m == "aa":
        return float(adamic_adar_score(None, u, v, cache=cache))
    if m == "cn":
        return float(common_neighbor_score(None, u, v, cache=cache))
    if m == "pa":
        return float(preferential_attachment_score(None, u, v, cache=cache))
    if m in ("hp", "hpi", "hub_promoted"):
        return float(hub_promoted_score(None, u, v, cache=cache))
    if m in ("lhn", "lhn1", "leicht_holme_newman"):
        return float(leicht_holme_newman_score(None, u, v, cache=cache))
    if m == "jaccard":
        return float(jaccard_score(None, u, v, cache=cache))
    if m in ("ra", "resource_allocation"):
        return float(resource_allocation_score(None, u, v, cache=cache))
    raise ValueError(f"Unknown method '{method}'")


def _compute_trial_metrics(
    candidate_pairs,
    scores,
    pos_edges,
    recall_grid,
):
    pos_edge_set = set(pos_edges)
    base_rate = (len(pos_edges) / len(candidate_pairs)) if candidate_pairs else None
    metrics = compute_metrics(
        candidate_pairs,
        scores,
        pos_edge_set,
        recall_grid,
    )
    labels = [1 if tuple(sorted(p)) in pos_edge_set else 0 for p in candidate_pairs]
    source_metrics = ranking_metrics_by_source(
        candidate_pairs,
        labels,
        scores,
        metrics.get("k_list", []),
        base_rate=base_rate,
    )
    metrics["source_ranking"] = source_metrics
    metrics["base_rate"] = float(base_rate) if base_rate is not None else float("nan")
    return metrics


def _aggregate_source_ranking(metrics_list: List[Dict]) -> Dict:
    buckets: Dict[str, List[float]] = {}
    for metrics in metrics_list:
        source_metrics = metrics.get("source_ranking", {})
        if not isinstance(source_metrics, dict):
            continue
        for key, value in source_metrics.items():
            if value is None:
                continue
            buckets.setdefault(str(key), []).append(float(value))
    return {key: float(np.nanmean(values)) for key, values in buckets.items()}


def _aggregate_method_metrics(metrics_list: List[Dict]) -> Dict:
    out = aggregate_metrics(metrics_list)
    if not out:
        return out
    out["base_rate"] = float(
        np.nanmean([m.get("base_rate", float("nan")) for m in metrics_list])
    )
    out["source_ranking"] = _aggregate_source_ranking(metrics_list)
    return out


def run_experiment(
    config_path: Path,
    out_dir: Path,
    neg_multiplier: int,
    variant_names: Sequence[str] | None,
    trials: int | None = None,
    max_workers: int = 0,
    max_edge_rows: int = MAX_EDGE_ROWS,
    resume: bool = True,
    task: str = TASK_WITHIN,
    cross_windows: Sequence[int] | None = None,
) -> Dict:
    cfg = json.loads(config_path.read_text(encoding="utf-8-sig"))
    trials = int(trials) if trials is not None else int(cfg.get("trials", 1))
    test_size = float(cfg.get("test_size", 0.5))
    recall_step = float(cfg.get("recall_step", 0.01))
    recall_end = float(cfg.get("recall_end", 0.1))
    base_seed = int(cfg.get("base_seed", 42))
    cfg_workers = int(cfg.get("max_workers", 0))
    if max_workers <= 0 and cfg_workers > 0:
        max_workers = cfg_workers
    if max_workers <= 0:
        max_workers = os.cpu_count() or 1

    weight_column_cfg = cfg.get("weight_column")
    if isinstance(weight_column_cfg, (list, tuple)):
        weight_column = weight_column_cfg[0]
    else:
        weight_column = weight_column_cfg

    recall_grid = build_recall_grid(step=recall_step, end=recall_end)

    variants = load_h3_variants(config_path)
    if variant_names:
        requested = {v.lower() for v in variant_names}
        variants = [v for v in variants if v.name.lower() in requested]
    else:
        default_variant = next(
            (v for v in variants if v.name.lower() == "norm_default"),
            None,
        )
        if default_variant is None:
            default_variant = next((v for v in variants if v.name.lower() == "default"), None)
        variants = [default_variant] if default_variant else variants[:1]

    unsup_methods = cfg.get("unsupervised", {}).get("methods", ["aa", "cn", "pa", "hp", "lhn", "jaccard", "ra"])
    method_list = ["l3"] + unsup_methods

    out_dir.mkdir(parents=True, exist_ok=True)
    all_outputs = {}

    task_norm = task.upper()
    if task_norm not in {TASK_WITHIN, TASK_CROSS_PERIOD}:
        raise ValueError(f"Unknown task '{task}' (expected A or B)")
    windows = list(cross_windows) if cross_windows else [90, 180]

    if task_norm == TASK_WITHIN:
        dataset_entries: List[Tuple[Path, Path | None, int | None]] = [
            (path, None, None) for path in _resolve_dataset_paths(max_edge_rows)
        ]
    elif task_norm == TASK_CROSS_PERIOD:
        dataset_entries = [
            (train_path, test_path, long_window)
            for train_path, test_path, long_window in _resolve_task_b_pairs(max_edge_rows, windows)
        ]
    for train_path, test_path, long_window in dataset_entries:
        if not train_path.is_file():
            raise FileNotFoundError(f"Dataset not found: {train_path}")
        if task_norm != TASK_WITHIN and (test_path is None or not test_path.is_file()):
            raise FileNotFoundError(f"Dataset not found: {test_path}")

        out_name = (
            f"{train_path.stem}_non_gnn.json"
            if task_norm == TASK_WITHIN
            else f"{train_path.stem}_to_{test_path.stem}_non_gnn_task_{task_norm}.json"
        )
        out_path = out_dir / out_name
        if resume and _is_complete_output(out_path, train_path, test_path, task_norm):
            print(f"[Skip] {train_path.name} -> {out_path} (already complete)")
            continue

        if task_norm == TASK_WITHIN:
            data = prepare_dataset(train_path, weight_col=weight_column)
            total_nodes = sorted(set(data["node1"]).union(set(data["node2"])))
            all_edges = set(_edge_set_from_df(data))
            train_df = data
            test_df = None
        else:
            train_df = prepare_dataset(train_path, weight_col=weight_column)
            test_df = prepare_dataset(test_path, weight_col=weight_column)
            total_nodes = sorted(
                set(train_df["node1"]).union(set(train_df["node2"]))
                .union(set(test_df["node1"]))
                .union(set(test_df["node2"]))
            )
            all_edges = set(_edge_set_from_df(test_df))

        dataset_payload = {
            "dataset": str(train_path),
            "task": task_norm,
            "neg_multiplier": int(neg_multiplier),
            "trials": trials,
            "test_size": test_size,
            "recall_step": recall_step,
            "recall_end": recall_end,
            "recall_grid": recall_grid,
            "methods": {},
        }
        if task_norm != TASK_WITHIN:
            dataset_payload["train_dataset"] = str(train_path)
            dataset_payload["test_dataset"] = str(test_path)
            dataset_payload["long_window"] = long_window

        method_buffers: Dict[str, Dict[str, List]] = {}
        for variant in variants:
            method_buffers[f"h3_{variant.name}"] = {"metrics": []}
        for method in method_list:
            method_buffers[method] = {"metrics": []}

        if task_norm == TASK_WITHIN:
            trials_iter = tqdm(range(trials), desc=f"{train_path.name} trials", unit="trial")
        else:
            trials_iter = tqdm(range(trials), desc=f"{train_path.name} -> {test_path.name} trials", unit="trial")

        train_edges_set = set(_edge_set_from_df(train_df))
        if task_norm != TASK_WITHIN and test_df is not None:
            test_edges_all = set(_edge_set_from_df(test_df))
            pos_edges_fixed = sorted(test_edges_all - train_edges_set)
        else:
            pos_edges_fixed = []

        for trial_idx in trials_iter:
            seed = base_seed + trial_idx
            set_all_seeds(seed)

            if task_norm == TASK_WITHIN:
                X_train, X_test = split_train_test(train_df, test_size=test_size, seed=seed)
                G_train = nx.Graph()
                for _, row in X_train.iterrows():
                    G_train.add_edge(row["node1"], row["node2"], weight=float(row["weight"]))
                G_train.add_nodes_from(total_nodes)
                cache = GraphCache.from_graph(G_train, total_nodes)

                pos_edges = [tuple(sorted(edge)) for edge in X_test[["node1", "node2"]].itertuples(index=False, name=None)]
                pos_edges = sorted(set(pos_edges))
                banned_edges = all_edges
                test_nodes = sorted(set(X_test["node1"]).union(set(X_test["node2"])))
            else:
                G_train = nx.Graph()
                for _, row in train_df.iterrows():
                    G_train.add_edge(row["node1"], row["node2"], weight=float(row["weight"]))
                G_train.add_nodes_from(total_nodes)
                cache = GraphCache.from_graph(G_train, total_nodes)

                pos_edges = list(pos_edges_fixed)
                banned_edges = all_edges
                test_nodes = sorted({n for edge in pos_edges for n in edge})

            if not pos_edges:
                empty_metrics = _compute_trial_metrics([], [], [], recall_grid)
                for payload in method_buffers.values():
                    payload["metrics"].append(empty_metrics)
                continue

            desired_neg = len(pos_edges) * int(neg_multiplier)
            neg_edges = _sample_distance2_negatives(
                G_train,
                test_nodes,
                banned_edges,
                desired_neg,
                seed=seed + 991,
            )
            candidate_pairs = pos_edges + neg_edges

            for variant in tqdm(variants, desc="H3 variants", leave=False):
                params = {
                    "forward_weight": variant.forward_weight,
                    "reverse_weight": variant.reverse_weight,
                    "penalty_gamma": variant.penalty_gamma,
                    "min_penalty": variant.min_penalty,
                    "connector_gamma": variant.connector_gamma,
                    "target_gamma": variant.target_gamma,
                    "path_weight_gamma": variant.path_weight_gamma,
                }
                scores = _score_pairs_parallel(
                    candidate_pairs,
                    max_workers,
                    _h3_worker_init,
                    (cache, params),
                    _h3_score_worker,
                    desc=f"H3 {variant.name}",
                    show_progress=False,
                )
                metrics = _compute_trial_metrics(candidate_pairs, scores, pos_edges, recall_grid)
                bucket = method_buffers[f"h3_{variant.name}"]
                bucket["metrics"].append(metrics)

            for method in tqdm(method_list, desc="Baselines", leave=False):
                scores = _score_pairs_parallel(
                    candidate_pairs,
                    max_workers,
                    _baseline_worker_init,
                    (cache, method),
                    _baseline_score_worker,
                    desc=f"Baseline {method}",
                    show_progress=True,
                )
                metrics = _compute_trial_metrics(candidate_pairs, scores, pos_edges, recall_grid)
                bucket = method_buffers[method]
                bucket["metrics"].append(metrics)

        for method_name, payload in method_buffers.items():
            dataset_payload["methods"][method_name] = _aggregate_method_metrics(payload["metrics"])

        out_path.write_text(json.dumps(dataset_payload, indent=2), encoding="utf-8")
        print(f"[Done] {train_path.name} -> {out_path}")

        all_outputs[str(train_path)] = dataset_payload

    return all_outputs


def _is_complete_output(
    out_path: Path,
    train_path: Path,
    test_path: Path | None = None,
    task: str | None = None,
) -> bool:
    if not out_path.is_file():
        return False
    try:
        payload = json.loads(out_path.read_text(encoding="utf-8-sig"))
    except Exception:
        return False
    if not isinstance(payload, dict):
        return False
    payload_task = payload.get("task")
    if task and payload_task and str(payload_task).upper() != str(task).upper():
        return False
    payload_dataset = payload.get("dataset")
    if payload_dataset:
        try:
            payload_name = Path(payload_dataset).name
        except Exception:
            payload_name = ""
        if str(train_path) != payload_dataset and train_path.name != payload_name:
            return False
    if test_path is not None:
        payload_test = payload.get("test_dataset")
        if payload_test:
            try:
                payload_test_name = Path(payload_test).name
            except Exception:
                payload_test_name = ""
            if str(test_path) != payload_test and test_path.name != payload_test_name:
                return False
    methods = payload.get("methods")
    return isinstance(methods, dict) and len(methods) > 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run H3 + non-GNN baselines on fixed datasets with 20x negatives."
    )
    parser.add_argument("--config", type=Path, default=Path("h3_config.json"))
    parser.add_argument("--out-dir", type=Path, default=Path("reports/taskA_new_non_gnn"))
    parser.add_argument("--neg-multiplier", type=int, default=20)
    parser.add_argument("--trials", type=int, default=None, help="Override number of trials.")
    parser.add_argument("--variants", nargs="*", default=None, help="Optional H3 variant names to run.")
    parser.add_argument("--max-workers", type=int, default=32, help="Process workers (default=32).")
    parser.add_argument("--max-edge-rows", type=int, default=MAX_EDGE_ROWS)
    parser.add_argument("--task", type=str, default=TASK_WITHIN, choices=[TASK_WITHIN, TASK_CROSS_PERIOD])
    parser.add_argument("--cross-windows", nargs="*", type=int, default=[30, 90, 180], help="Long windows for task B.")
    parser.add_argument("--no-resume", action="store_true")
    args = parser.parse_args()

    task_norm = args.task.upper()
    run_experiment(
        args.config,
        args.out_dir,
        args.neg_multiplier,
        args.variants,
        args.trials,
        args.max_workers,
        max_edge_rows=args.max_edge_rows,
        resume=(not args.no_resume),
        task=args.task,
        cross_windows=args.cross_windows,
    )


if __name__ == "__main__":
    main()

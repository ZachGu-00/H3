"""
Core H3 implementation and shared utilities.

This module centralizes:
- GraphCache for quick neighbor/degree/weight lookups.
- The weighted/bidirectional H3 scorer.
- Lightweight helpers used by both unsupervised and supervised runs.
"""

from __future__ import annotations

import json
import math
import random
import os
import concurrent.futures
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple, Union

import networkx as nx
import numpy as np
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.metrics import build_recall_grid, compute_metrics, aggregate_metrics
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# -------------------- Config & seed helpers -------------------- #


@dataclass
class H3Variant:
    name: str
    forward_weight: float = 0.6
    reverse_weight: float = 0.4
    penalty_gamma: float = 0.8
    min_penalty: float = 1.0
    connector_gamma: float = 0.5
    target_gamma: float = 0.5
    path_weight_gamma: float = 1 / 3


def load_h3_variants(config_path: Path) -> List[H3Variant]:
    """Load H3 variants from the shared JSON config."""
    with config_path.open("r", encoding="utf-8-sig") as f:
        cfg = json.load(f)
    variants = []
    for entry in cfg.get("h3_variants", []):
        variants.append(H3Variant(**entry))
    if not variants:
        variants.append(H3Variant(name="default"))
    return variants


def _resolve_h3_datasets(dataset_entry: Union[str, Sequence[str]], pattern: str = "*_30.csv") -> List[Path]:
    """
    Resolve dataset inputs for H3 sweeps.

    - If a file is provided, returns that file.
    - If a directory is provided, glob with pattern (default *_30.csv) and sort.
    - If a list is provided, resolve each entry.
    """
    paths: List[Path] = []
    if isinstance(dataset_entry, (list, tuple)):
        entries = dataset_entry
    else:
        entries = [dataset_entry]

    for entry in entries:
        p = Path(entry)
        if p.is_file():
            paths.append(p.resolve())
        elif p.is_dir():
            matched = sorted([x for x in p.glob(pattern) if x.is_file()])
            if not matched:
                raise FileNotFoundError(f"No CSV files matched {pattern} under directory {p}")
            paths.extend([x.resolve() for x in matched])
        else:
            raise FileNotFoundError(f"Dataset path not found: {p}")
    return paths


def resolve_h3_datasets(dataset_entry: Union[str, Sequence[str]], pattern: str = "*_30.csv") -> List[Path]:
    """Public wrapper for resolving dataset inputs consistently across runners."""
    return _resolve_h3_datasets(dataset_entry, pattern=pattern)


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


# -------------------- Graph utilities -------------------- #


class GraphCache:
    """Cache graph metadata to speed up repeated scoring calls.

    稀疏版实现：使用 dict-of-dicts 存储邻接表，避免 N×N dense 矩阵占用巨量内存。
    """

    def __init__(
        self,
        node_list: List,
        node_to_idx: Dict,
        neighbor_sets: Dict,
        degree_array: np.ndarray,
        weighted_degree_array: np.ndarray,
        adjacency_dict: Dict,
        self_loops: set,
    ):
        self.node_list = node_list
        self.node_to_idx = node_to_idx
        self.neighbor_sets = neighbor_sets
        self.degree_array = degree_array
        self.weighted_degree_array = weighted_degree_array

        # 稀疏邻接：u -> {v: weight}
        self.adjacency = adjacency_dict

        self.self_loops = self_loops

    @classmethod
    def from_graph(cls, G: nx.Graph, total_nodes: Iterable) -> "GraphCache":
        node_list = sorted(total_nodes)
        node_to_idx = {node: idx for idx, node in enumerate(node_list)}
        num_nodes = len(node_list)

        neighbor_sets: Dict = {}
        degree_array = np.zeros(num_nodes, dtype=np.float32)
        weighted_degree_array = np.zeros(num_nodes, dtype=np.float32)

        # 初始化稀疏邻接：只为出现在 node_list 中的节点建 dict
        adjacency: Dict = {node: {} for node in node_list}
        self_loops = set()

        # 构建邻接表 & self-loop
        for u, v, data in G.edges(data=True):
            if u not in node_to_idx or v not in node_to_idx:
                continue
            w = float(data.get("weight", 1.0))

            # 无向图：两边都写
            adjacency[u][v] = w
            adjacency[v][u] = w

            if u == v:
                self_loops.add(u)

        # 构建邻居集合 & 度 / 加权度
        for node in node_list:
            idx = node_to_idx[node]
            if node in adjacency:
                neighbors = frozenset(adjacency[node].keys())
                weight_sum = float(sum(adjacency[node].values()))
            else:
                neighbors = frozenset()
                weight_sum = 0.0

            neighbor_sets[node] = neighbors
            degree_array[idx] = float(len(neighbors))
            weighted_degree_array[idx] = weight_sum

        return cls(
            node_list=node_list,
            node_to_idx=node_to_idx,
            neighbor_sets=neighbor_sets,
            degree_array=degree_array,
            weighted_degree_array=weighted_degree_array,
            adjacency_dict=adjacency,
            self_loops=self_loops,
        )

    def neighbors(self, node):
        return self.neighbor_sets.get(node, frozenset())

    def degree(self, node):
        idx = self.node_to_idx.get(node)
        if idx is None:
            return 0.0
        return float(self.degree_array[idx])

    def weighted_degree(self, node):
        idx = self.node_to_idx.get(node)
        if idx is None:
            return 0.0
        return float(self.weighted_degree_array[idx])

    def has_edge(self, u, v):
        return v in self.adjacency.get(u, ())

    def edge_weight(self, u, v):
        return float(self.adjacency.get(u, {}).get(v, 0.0))

    def has_self_loop(self, node):
        return node in self.self_loops


# -------------------- Scoring functions -------------------- #


def l_score(
    G,
    x,
    y,
    cache: Optional[GraphCache] = None,
    path_len: int = 3,
):
    """
    L2/L3 score with optional cache support.
    """
    if cache is None:
        if x not in G or y not in G:
            return 0.0
        neighbor_fn = G.neighbors
        has_edge = G.has_edge
        degree_fn = G.degree
        has_self_loop = lambda node: G.has_edge(node, node)
    else:
        if x not in cache.node_to_idx or y not in cache.node_to_idx:
            return 0.0
        neighbor_fn = cache.neighbors
        has_edge = cache.has_edge
        degree_fn = cache.degree
        has_self_loop = cache.has_self_loop

    if path_len not in (2, 3):
        raise ValueError("L-score supports only length-2 or length-3 paths.")

    score = 0.0
    if path_len == 2:
        deg_y = degree_fn(y)
        kY = deg_y - 1 if has_self_loop(y) else deg_y
        if kY <= 0:
            return 0.0
        for U in neighbor_fn(x):
            if has_edge(U, y):
                deg_u = degree_fn(U)
                kU = deg_u - 1 if has_self_loop(U) else deg_u
                if kU > 0:
                    score += 1.0 / math.sqrt(kU * kY)
        return score

    for U in neighbor_fn(x):
        for V in neighbor_fn(U):
            if has_edge(V, y):
                deg_u = degree_fn(U)
                deg_v = degree_fn(V)
                kU = deg_u - 1 if has_self_loop(U) else deg_u
                kV = deg_v - 1 if has_self_loop(V) else deg_v
                if kU > 0 and kV > 0:
                    score += 1.0 / math.sqrt(kU * kV)
    return score


def h3_score(
    G,
    x,
    y,
    cache: Optional[GraphCache] = None,
    forward_weight: float = 0.6,
    reverse_weight: float = 0.4,
    penalty_gamma: float = 0.8,
    min_penalty: float = 1.0,
    connector_gamma: float = 0.5,
    target_gamma: float = 0.5,
    path_weight_gamma: float = 1 / 3,
):
    """
    Weighted/bidirectional H3 with tunable diversity penalty (log-count^gamma).

    - Penalizes crowded intermediary paths through log(diversity)^gamma.
    - Uses weighted edges and degree-based normalization on intermediaries/targets.
    """
    if cache is None:
        if x not in G or y not in G:
            return 0.0

        neighbor_fn = G.neighbors

        def weighted_degree(node):
            return sum(G[node][nbr].get("weight", 1.0) for nbr in G.neighbors(node))

        def edge_weight(u, v):
            if not G.has_edge(u, v):
                return 0.0
            return G[u][v].get("weight", 1.0)
    else:
        if x not in cache.node_to_idx or y not in cache.node_to_idx:
            return 0.0
        neighbor_fn = cache.neighbors
        weighted_degree = cache.weighted_degree
        edge_weight = cache.edge_weight

    neighbor_cache = {}

    def neighbors(node):
        if node in neighbor_cache:
            return neighbor_cache[node]
        if cache is None:
            if node not in G:
                result = frozenset()
            else:
                result = frozenset(neighbor_fn(node))
        else:
            result = neighbor_fn(node)
        neighbor_cache[node] = result
        return result

    def _diversity_penalty(count: int) -> float:
        base = math.log1p(count)
        penalty = math.pow(base, penalty_gamma) if base > 0 else 1.0
        return max(penalty, min_penalty)

    def _directional_score(source, target):
        source_neighbors = neighbors(source)
        target_neighbors = neighbors(target)
        if not source_neighbors or not target_neighbors:
            return 0.0

        target_norm = math.pow(max(weighted_degree(target), 1.0), target_gamma)
        total = 0.0
        for mid in source_neighbors:
            mid_neighbors = neighbors(mid)
            if not mid_neighbors:
                continue
            connectors = mid_neighbors & target_neighbors
            if not connectors:
                continue

            w_source_mid = edge_weight(source, mid)
            k_mid = weighted_degree(mid)
            if k_mid <= 0 or w_source_mid <= 0:
                continue

            penalty = _diversity_penalty(len(connectors))
            mid_norm = math.pow(max(k_mid, 1.0), connector_gamma)
            for connector in connectors:
                w_mid_connector = edge_weight(mid, connector)
                w_conn_target = edge_weight(connector, target)
                if w_mid_connector <= 0 or w_conn_target <= 0:
                    continue
                k_conn = weighted_degree(connector)
                if k_conn <= 0:
                    continue
                conn_norm = math.pow(max(k_conn, 1.0), connector_gamma)
                path_strength = w_source_mid * w_mid_connector * w_conn_target
                if path_strength <= 0:
                    continue
                scaled_path = math.pow(max(path_strength, 1e-12), path_weight_gamma)
                total += scaled_path / (mid_norm * conn_norm * target_norm * penalty)
        return total

    forward = _directional_score(x, y)
    reverse = _directional_score(y, x)
    base_score = forward_weight * forward + reverse_weight * reverse

    return base_score


# -------------------- Dataset helpers -------------------- #


def prepare_dataset(csv_path, node1_col=None, node2_col=None, weight_col=None, split_col=None):
    """Load dataset and standardize column names.

    If available, retains auxiliary edge features (paircount/benecount/samedaycount)
    so downstream models can use them. Weight column can be forced via weight_col;
    otherwise we prefer benecount > paircount > weight-like fallbacks.
    """
    import pandas as pd

    df = pd.read_csv(csv_path)

    def _resolve_column(explicit_name, candidates, role):
        search_order = []
        if explicit_name:
            search_order.append(explicit_name)
        search_order.extend(candidates)
        seen = set()
        for name in search_order:
            if not name or name in seen:
                continue
            seen.add(name)
            if name in df.columns:
                return name
        raise ValueError(f"Could not locate a column for {role} in {csv_path}")

    node1_candidates = [
        "node1",
        "npi1",
        "src",
        "source",
        "source_id",
        "source_node",
        "from",
        "from_id",
        "provider1",
        "provider_source",
    ]
    node2_candidates = [
        "node2",
        "npi2",
        "dst",
        "dest",
        "destination",
        "target",
        "target_id",
        "to",
        "to_id",
        "provider2",
        "provider_target",
    ]
    weight_candidates = [
        "benecount",
        "paircount",
        "weight",
        "weight_value",
        "edge_weight",
        "samedaycount",
        "count",
        "frequency",
        "value",
    ]

    node1_name = _resolve_column(node1_col, node1_candidates, "node1")
    node2_name = _resolve_column(node2_col, node2_candidates, "node2")
    weight_name = _resolve_column(weight_col, weight_candidates, "weight")

    columns = [node1_name, node2_name, weight_name]
    extra_cols = [extra for extra in ("paircount", "benecount", "samedaycount") if extra in df.columns and extra not in columns]
    if split_col and split_col in df.columns:
        columns.append(split_col)
    df = df[columns + extra_cols].copy()
    rename_map = {node1_name: "node1", node2_name: "node2", weight_name: "weight"}
    for extra in extra_cols:
        rename_map.setdefault(extra, extra)
    if split_col and split_col in df.columns:
        rename_map[split_col] = "split"
    df = df.rename(columns=rename_map)
    return df


def build_candidate_pairs(G: nx.Graph, total_nodes: Iterable, required_edges: Sequence[Tuple]) -> List[Tuple]:
    """
    Generate a reduced set of candidate node pairs for scoring.

    - Includes all two-hop pairs observed in the training graph.
    - Ensures every evaluation edge (from X_test) is present.
    """
    candidate_pairs = set()
    adjacency = {node: set(G.neighbors(node)) for node in G.nodes()}

    for node in total_nodes:
        adjacency.setdefault(node, set())

    for u, neighbors in adjacency.items():
        for nbr in neighbors:
            for target in adjacency.get(nbr, ()):
                if target == u or G.has_edge(u, target):
                    continue
                pair = (u, target) if u < target else (target, u)
                candidate_pairs.add(pair)

    for u, v in required_edges:
        pair = (u, v) if u < v else (v, u)
        if not G.has_edge(*pair):
            candidate_pairs.add(pair)

    return sorted(candidate_pairs)


def limit_candidate_pairs(
    candidate_pairs: List[Tuple],
    required_edges: Sequence[Tuple],
    max_candidates: Optional[int],
    seed: int = 42,
) -> List[Tuple]:
    """
    Limit candidate pairs while keeping all required_edges.

    - required_edges are always kept.
    - If max_candidates is None/<=0 or list is already small enough, return as-is.
    - Otherwise randomly sample non-required pairs (deterministic by seed) to fit the cap.
    """
    if not max_candidates or max_candidates <= 0 or len(candidate_pairs) <= max_candidates:
        return candidate_pairs

    required_set = {tuple(sorted(e)) for e in required_edges}
    required_pairs = [p for p in candidate_pairs if p in required_set]
    if len(required_pairs) >= max_candidates:
        return required_pairs

    non_required = [p for p in candidate_pairs if p not in required_set]
    rng = random.Random(seed)
    rng.shuffle(non_required)
    quota = max(0, max_candidates - len(required_pairs))
    sampled = non_required[:quota]
    return required_pairs + sampled


def compute_candidate_cap(
    candidate_pairs_len: int,
    required_edges_len: int,
    train_edges_len: int,
    cfg_value,
    auto_factor: float = 10.0,
    auto_ceiling: Optional[int] = None,
) -> Optional[int]:
    """
    Resolve candidate cap from config.

    - If cfg_value is None -> no cap.
    - If cfg_value == "auto" -> scale with train_edges_len * auto_factor,
      always >= required_edges_len, optionally bounded by auto_ceiling,
      and never exceeds actual candidate_pairs_len.
    - If cfg_value is numeric -> fixed cap.
    """
    if cfg_value is None:
        return None
    if isinstance(cfg_value, str) and cfg_value.lower() == "auto":
        cap = max(required_edges_len, int(train_edges_len * auto_factor))
        if auto_ceiling and auto_ceiling > 0:
            cap = min(cap, int(auto_ceiling))
        return min(candidate_pairs_len, cap)
    return int(cfg_value)


def sample_candidate_pairs(
    G: nx.Graph,
    candidate_pairs: List[Tuple],
    required_edges: Sequence[Tuple],
    cap_value,
    seed: int,
    auto_factor: float = 3.0,
    auto_ceiling: Optional[int] = 10_000_000,
    quantiles: Sequence[float] = (50.0, 90.0),
    bucket_weights: Sequence[float] = (0.3, 0.5, 0.2),
    min_per_node: int = 0,
) -> List[Tuple]:
    """
    Apply adaptive/bucketed sampling to candidate pairs while keeping required edges.

    - cap_value: None (no cap), number, or "auto".
    - "auto": cap = log1p(|E_train|) * auto_factor * max(|E_train|, 1) (ceil), clipped by auto_ceiling.
    - Bucket sampling by degree percentiles (quantiles), with per-bucket weights.
    - Optional node coverage: ensure each node appears in at least min_per_node negative pairs if possible.
    """
    required_set = {tuple(sorted(e)) for e in required_edges}
    required_pairs = [p for p in candidate_pairs if p in required_set]
    if cap_value is None:
        return candidate_pairs

    train_edges_len = G.number_of_edges()
    total_len = len(candidate_pairs)

    if isinstance(cap_value, str) and cap_value.lower() == "auto":
        cap = math.ceil(math.log1p(train_edges_len) * auto_factor * max(train_edges_len, 1))
        if auto_ceiling and auto_ceiling > 0:
            cap = min(cap, int(auto_ceiling))
        cap = max(cap, len(required_pairs))
    else:
        cap = max(int(cap_value), len(required_pairs))

    cap = min(cap, total_len)
    if len(required_pairs) >= cap:
        return required_pairs

    non_required = [p for p in candidate_pairs if p not in required_set]
    if len(non_required) <= cap - len(required_pairs):
        return required_pairs + non_required

    # Compute degree metric per pair (max degree of the endpoints).
    deg = dict(G.degree())

    def deg_metric(pair):
        return max(deg.get(pair[0], 0), deg.get(pair[1], 0))

    metrics = np.array([deg_metric(p) for p in non_required], dtype=np.float64)
    # Determine bucket edges
    quantiles = list(quantiles) if quantiles else [50.0, 90.0]
    bucket_edges = [np.percentile(metrics, q) for q in quantiles]

    buckets = [[] for _ in range(len(bucket_edges) + 1)]
    for pair, m in zip(non_required, metrics):
        idx = 0
        while idx < len(bucket_edges) and m > bucket_edges[idx]:
            idx += 1
        buckets[idx].append(pair)

    # Normalize weights
    if not bucket_weights or len(bucket_weights) != len(buckets):
        bucket_weights = [1.0] * len(buckets)
    weight_sum = sum(bucket_weights)
    bucket_weights = [w / weight_sum for w in bucket_weights]

    rng = random.Random(seed)
    remaining_quota = cap - len(required_pairs)
    sampled: List[Tuple] = []

    # Sample per bucket
    for b_pairs, w in zip(buckets, bucket_weights):
        if remaining_quota <= 0 or not b_pairs:
            continue
        quota = int(round(remaining_quota * w))
        quota = min(quota, len(b_pairs))
        rng.shuffle(b_pairs)
        sampled.extend(b_pairs[:quota])
        remaining_quota -= quota

    # If quota remains, fill from leftover
    if remaining_quota > 0:
        leftovers = []
        for b_pairs in buckets:
            leftovers.extend(b_pairs)
        rng.shuffle(leftovers)
        needed = min(remaining_quota, len(leftovers))
        sampled.extend(leftovers[:needed])
        remaining_quota -= needed

    # Optional: ensure per-node coverage in negative samples
    if min_per_node > 0 and remaining_quota == 0:
        counts = {}
        for u, v in sampled:
            counts[u] = counts.get(u, 0) + 1
            counts[v] = counts.get(v, 0) + 1
        # Build leftover pool excluding already sampled
        sampled_set = set(sampled)
        pool = [p for p in non_required if p not in sampled_set]
        rng.shuffle(pool)
        pool_iter = iter(pool)
        added = []
        for node in deg:
            need = min_per_node - counts.get(node, 0)
            while need > 0:
                try:
                    p = next(pool_iter)
                except StopIteration:
                    break
                if node not in p:
                    continue
                added.append(p)
                sampled_set.add(p)
                counts[p[0]] = counts.get(p[0], 0) + 1
                counts[p[1]] = counts.get(p[1], 0) + 1
                need -= 1
            if len(required_pairs) + len(sampled) + len(added) >= cap:
                break
        sampled.extend(added)

    # Final trim (safety)
    if len(sampled) > cap - len(required_pairs):
        sampled = sampled[: cap - len(required_pairs)]

    return required_pairs + sampled


# -------------------- Parallel helpers for H3 scoring -------------------- #

_H3_WORKER_STATE = {"cache": None, "params": None}


def _h3_worker_init(cache: GraphCache, params: Dict):
    """Initializer to stash shared cache/params in each worker to avoid re-pickling per task."""
    _H3_WORKER_STATE["cache"] = cache
    _H3_WORKER_STATE["params"] = params


def _h3_score_worker(pair):
    cache = _H3_WORKER_STATE["cache"]
    params = _H3_WORKER_STATE["params"]
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


# -------------------- Train/test split helper -------------------- #


def split_train_test(df, test_size=0.5, seed=42):
    """Split a standardized dataframe into train/test components."""
    # be robust to datasets whose weight column hasn't been renamed
    weight_col = "weight"
    if weight_col not in df.columns:
        for cand in ["benecount", "paircount", "weight_value", "edge_weight", "count", "frequency", "value"]:
            if cand in df.columns:
                weight_col = cand
                break
    X = df[["node1", "node2", weight_col]]
    y = df[weight_col]
    X_train, X_test, _, _ = train_test_split(X, y, test_size=test_size, random_state=seed)
    return X_train, X_test


# -------------------- H3-only experiment runner -------------------- #


def run_h3_variants(
    config_path: Path = Path("h3_config.json"),
    variant_names: Optional[Sequence[str]] = None,
    dataset_override: Optional[Sequence[Union[Path, str]]] = None,
    skip_existing: bool = False,
) -> Dict:
    """Run H3 sweeps based on the shared config.

    Args:
        config_path: Path to config JSON.
        variant_names: Optional subset of variant names to run. If None, run all variants in config.
    """
    cfg_path = config_path.resolve()
    with cfg_path.open("r", encoding="utf-8") as f:
        cfg = json.load(f)

    dataset_entry = cfg["dataset"]
    dataset_pattern = cfg.get("dataset_pattern", "*_30.csv")
    weight_column_cfg = cfg.get("weight_column") or "benecount"
    if isinstance(weight_column_cfg, (list, tuple)):
        weight_column = weight_column_cfg[0]
    else:
        weight_column = weight_column_cfg
    trials = int(cfg.get("trials", 10))
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
    output_base = Path(cfg.get("output", "reports_h3/h3_variants.json"))
    output_is_file = bool(output_base.suffix)
    output_dir = output_base.parent if output_is_file else output_base
    max_workers = int(cfg.get("h3_max_workers", 0))
    if max_workers <= 0:
        max_workers = os.cpu_count() or 1

    variants = load_h3_variants(cfg_path)
    if variant_names:
        name_set = set(variant_names)
        variants = [v for v in variants if v.name in name_set]
        if not variants:
            raise ValueError(f"No H3 variants matched requested names: {variant_names}")
    recall_grid = build_recall_grid(step=recall_step, end=recall_end)
    if dataset_override:
        dataset_paths = [Path(p).resolve() for p in dataset_override]
    else:
        dataset_paths = resolve_h3_datasets(dataset_entry, pattern=dataset_pattern)

    # When dataset_override is used or multiple datasets are resolved, split per-dataset outputs
    # even if the config points to a single file, to avoid clobbering between runs.
    force_split_outputs = bool(dataset_override) or len(dataset_paths) > 1

    all_outputs = {}
    for dataset_path in dataset_paths:
        if output_is_file and not force_split_outputs:
            dataset_output_path = output_base
        else:
            output_dir.mkdir(parents=True, exist_ok=True)
            if output_is_file:
                dataset_output_path = output_dir / f"{dataset_path.stem.lower()}_{output_base.stem}{output_base.suffix}"
            else:
                dataset_output_path = output_dir / f"{dataset_path.stem.lower()}_h3_variants.json"

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

            if dataset_output_path.is_file():
                existing_path = dataset_output_path
                try:
                    existing_payload = json.loads(dataset_output_path.read_text(encoding="utf-8-sig"))
                except Exception as exc:  # pragma: no cover - defensive fallback
                    print(f"[Warn] Failed to read existing H3 file, recomputing: {exc}")
            elif output_is_file and output_base.is_file():
                # Older runs may have combined all datasets into a single file.
                try:
                    base_json = json.loads(output_base.read_text(encoding="utf-8-sig"))
                    if isinstance(base_json, dict):
                        if _matches_dataset(base_json.get("dataset")) or base_json.get("dataset") == str(dataset_path):
                            existing_payload = base_json
                            existing_path = output_base
                        elif str(dataset_path) in base_json:
                            existing_payload = base_json[str(dataset_path)]
                            existing_path = output_base
                        else:
                            # Fallback: try to find by matching stem/name among keys
                            for key, val in base_json.items():
                                if _matches_dataset(key):
                                    existing_payload = val
                                    existing_path = output_base
                                    break
                except Exception as exc:  # pragma: no cover - defensive fallback
                    print(f"[Warn] Failed to read combined H3 file, recomputing: {exc}")
            if existing_payload is not None:
                print(f"[Skip] H3 existing results found at {existing_path}")
                all_outputs[str(dataset_path)] = existing_payload
                continue

        print(f"[Start] H3 | {dataset_path.name} | weight={weight_column}")
        all_data = prepare_dataset(dataset_path, weight_col=weight_column)
        if "weight" not in all_data.columns:
            if weight_column in all_data.columns:
                all_data = all_data.copy()
                all_data["weight"] = all_data[weight_column].astype(float)
            else:
                raise KeyError(f"No weight column found (expected '{weight_column}') in {dataset_path}")
        total_nodes = sorted(set(all_data["node1"]).union(set(all_data["node2"])))

        experiments_output = []
        outer = tqdm(variants, desc="H3 configs", unit="cfg")
        for variant in outer:
            trial_metrics: List[Dict] = []
            meta_rows: List[Dict] = []
            inner = tqdm(range(trials), desc=f"{variant.name} trials", leave=False)
            for t in inner:
                set_all_seeds(base_seed + t)
                X_train, X_test = split_train_test(all_data, test_size=test_size, seed=base_seed + t)

                G_train = nx.Graph()
                for _, row in X_train.iterrows():
                    w = float(row["weight"]) if "weight" in row else float(row.get(weight_column, 1.0))
                    G_train.add_edge(row["node1"], row["node2"], weight=w)
                G_train.add_nodes_from(total_nodes)

                test_edges = [tuple(edge) for edge in X_test[["node1", "node2"]].itertuples(index=False, name=None)]
                test_edge_set = {tuple(sorted(edge)) for edge in test_edges}
                candidate_pairs = build_candidate_pairs(G_train, total_nodes, test_edges)
                candidate_pairs = sample_candidate_pairs(
                    G_train,
                    candidate_pairs,
                    test_edges,
                    cap_value=max_candidates_cfg,
                    seed=base_seed + t,
                    auto_factor=auto_factor,
                    auto_ceiling=auto_ceiling,
                    quantiles=bucket_quantiles,
                    bucket_weights=bucket_weights,
                    min_per_node=min_per_node,
                )
                cache = GraphCache.from_graph(G_train, total_nodes)

                variant_params = {
                    "forward_weight": variant.forward_weight,
                    "reverse_weight": variant.reverse_weight,
                    "penalty_gamma": variant.penalty_gamma,
                    "min_penalty": variant.min_penalty,
                    "connector_gamma": variant.connector_gamma,
                    "target_gamma": variant.target_gamma,
                    "path_weight_gamma": variant.path_weight_gamma,
                }

                score_desc = f"{dataset_path.name} | {variant.name} | trial {t + 1} score"
                if max_workers > 1 and len(candidate_pairs) > 0:
                    with concurrent.futures.ProcessPoolExecutor(
                        max_workers=max_workers,
                        initializer=_h3_worker_init,
                        initargs=(cache, variant_params),
                    ) as executor:
                        scores = []
                        with tqdm(total=len(candidate_pairs), desc=score_desc, leave=False) as pbar:
                            for s in executor.map(_h3_score_worker, candidate_pairs, chunksize=512):
                                scores.append(s)
                                pbar.update(1)
                else:
                    iterator = candidate_pairs
                    scores = []
                    bar = None
                    if len(candidate_pairs) > 0:
                        bar = tqdm(candidate_pairs, desc=score_desc, leave=False)
                        iterator = bar
                    scores = [
                        h3_score(
                            None,
                            u,
                            v,
                            cache=cache,
                            forward_weight=variant.forward_weight,
                            reverse_weight=variant.reverse_weight,
                            penalty_gamma=variant.penalty_gamma,
                            min_penalty=variant.min_penalty,
                            connector_gamma=variant.connector_gamma,
                            target_gamma=variant.target_gamma,
                            path_weight_gamma=variant.path_weight_gamma,
                        )
                        for u, v in iterator
                    ]
                    if bar:
                        bar.close()
                metrics = compute_metrics(
                    candidate_pairs,
                    scores,
                    test_edge_set,
                    recall_grid,
                    total_nodes=total_nodes,
                )
                trial_metrics.append(metrics)
                meta_rows.append(
                    {
                        "train_size": len(X_train),
                        "test_size": len(X_test),
                        "candidate_pairs": len(candidate_pairs),
                    }
                )
                inner.set_postfix(last_mean=f"{metrics['precision_at_recall_mean']:.4f}")
            inner.close()

            stats = aggregate_metrics(trial_metrics)
            experiments_output.append(
                {
                    "name": variant.name,
                    "params": {
                        "forward_weight": variant.forward_weight,
                        "reverse_weight": variant.reverse_weight,
                        "penalty_gamma": variant.penalty_gamma,
                        "min_penalty": variant.min_penalty,
                        "connector_gamma": variant.connector_gamma,
                        "target_gamma": variant.target_gamma,
                        "path_weight_gamma": variant.path_weight_gamma,
                    },
                    "metrics": stats,
                    "trial_meta": meta_rows,
                }
            )
        outer.close()

        output_payload = {
            "dataset": str(dataset_path),
            "trials": trials,
            "test_size": test_size,
            "recall_step": recall_step,
            "recall_end": recall_end,
            "recall_grid": recall_grid,
            "weight_column": weight_column,
            "experiments": experiments_output,
        }
        if output_is_file and not force_split_outputs:
            dataset_output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
            print(f"[Done] H3 sweeps finished | {dataset_path.name}")
            for exp in experiments_output:
                overall = exp["metrics"]["overall_mean"]
                print(f"  {exp['name']}: {overall:.6f}")
        else:
            dataset_output_path.write_text(json.dumps(output_payload, indent=2), encoding="utf-8")
            print(f"[Done] H3 sweeps saved to {dataset_output_path}")
            for exp in experiments_output:
                overall = exp["metrics"]["overall_mean"]
                print(f"  {exp['name']}: {overall:.6f}")
        all_outputs[str(dataset_path)] = output_payload

    # Persist combined output when a file path (default) is provided.
    if output_is_file and not force_split_outputs:
        output_dir.mkdir(parents=True, exist_ok=True)
        combined_payload = next(iter(all_outputs.values())) if len(all_outputs) == 1 else all_outputs
        output_base.write_text(json.dumps(combined_payload, indent=2), encoding="utf-8")
        print(f"[Done] H3 sweeps saved to {output_base}")

    # Return payloads (if multiple datasets, return a dict keyed by path)
    if len(all_outputs) == 1:
        return next(iter(all_outputs.values()))
    return all_outputs


__all__ = [
    "GraphCache",
    "H3Variant",
    "build_candidate_pairs",
    "h3_score",
    "l_score",
    "load_h3_variants",
    "compute_candidate_cap",
    "limit_candidate_pairs",
    "sample_candidate_pairs",
    "resolve_h3_datasets",
    "prepare_dataset",
    "run_h3_variants",
    "set_all_seeds",
    "split_train_test",
]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run H3 variant sweeps.")
    parser.add_argument("--config", type=Path, default=Path("h3_config.json"), help="Path to config JSON.")
    parser.add_argument(
        "--variants",
        nargs="*",
        default=None,
        help="Optional subset of variant names to run.",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip datasets whose outputs already exist.",
    )
    args = parser.parse_args()

    run_h3_variants(args.config, variant_names=args.variants, skip_existing=args.resume)

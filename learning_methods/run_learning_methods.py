from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
import torch
import sys
from pathlib import Path as PathLib
from sklearn.model_selection import train_test_split
from torch_geometric.utils import negative_sampling
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(PathLib(__file__).parent.parent))

from learning_methods.GNN import (
    GAT,
    GCN,
    GIN,
    GraphConvNet,
    GraphSAGE,
    build_node2vec_features,
    evaluate_gnn,
    train_gnn,
)
from h3.h3_core import prepare_dataset, set_all_seeds
from utils.metrics import build_recall_grid, compute_metrics, aggregate_metrics
from learning_methods.run_walk_embeddings import (
    resolve_dataset_paths,
    run as run_walk_embeddings,
    _evaluate_embeddings,
    _sample_distance2_negatives,
    _train_embeddings,
)


MAX_EDGE_ROWS = 600_000
TASK_WITHIN = "A"
TASK_CROSS_PERIOD = "B"


@dataclass(frozen=True)
class TaskPair:
    train_path: Path
    test_path: Path
    label: str
    long_window: int | None = None


def _count_edges(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8") as f:
        next(f, None)
        return sum(1 for _ in f)


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
    for root in [Path("data/2014data"), Path("data/2015data")]:
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
) -> List[TaskPair]:
    index = _index_datasets()
    pairs: List[TaskPair] = []
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
            label = f"{train_path.stem}_to_{test_path.stem}"
            pairs.append(TaskPair(train_path, test_path, label, long_window))
    return pairs


def _node2vec_cache_path(
    cache_dir: Path,
    pair: TaskPair,
    seed: int,
    task: str,
) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{pair.label}_seed{seed}_task{task}_node2vec_walk.npz"


def _load_node2vec_cache(cache_path: Path, nodes: List) -> torch.Tensor | None:
    if not cache_path.is_file():
        return None
    cached = np.load(cache_path, allow_pickle=True)
    cached_nodes = list(cached["nodes"])
    if cached_nodes != nodes:
        return None
    return torch.tensor(cached["embeddings"], dtype=torch.float)


def _prepare_pyg_data_cross(
    G: nx.Graph,
    train_edges: Sequence[Tuple],
    test_edges: Sequence[Tuple],
    nodes: List,
    node_features: torch.Tensor | None,
    test_neg_multiplier: int,
    banned_edge_set: set[Tuple],
) -> Tuple["torch_geometric.data.Data", torch.Tensor, torch.Tensor]:
    from torch_geometric.data import Data

    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    if node_features is None:
        x = torch.eye(len(nodes), dtype=torch.float)
    else:
        x = node_features

    edge_index = []
    edge_weight = []
    for u, v in train_edges:
        if u not in node_to_idx or v not in node_to_idx:
            continue
        idx_u = node_to_idx[u]
        idx_v = node_to_idx[v]
        edge_index.append([idx_u, idx_v])
        edge_index.append([idx_v, idx_u])
        w = G[u][v].get("weight", 1.0)
        edge_weight.append(w)
        edge_weight.append(w)

    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)

    test_edge_index = []
    test_edge_label = []

    pos_edges = set()
    for u, v in test_edges:
        if u not in node_to_idx or v not in node_to_idx:
            continue
        idx_u = node_to_idx[u]
        idx_v = node_to_idx[v]
        if idx_u == idx_v:
            continue
        pair = tuple(sorted([idx_u, idx_v]))
        if pair not in pos_edges:
            pos_edges.add(pair)
            test_edge_index.append([idx_u, idx_v])
            test_edge_label.append(1)

    num_neg = len(pos_edges) * max(1, int(test_neg_multiplier))
    test_nodes = sorted({n for edge in test_edges for n in edge})
    neg_pairs = _sample_distance2_negatives(
        G,
        test_nodes,
        banned_edge_set,
        num_neg,
        seed=42,
    )
    for u, v in neg_pairs:
        if u not in node_to_idx or v not in node_to_idx:
            continue
        test_edge_index.append([node_to_idx[u], node_to_idx[v]])
        test_edge_label.append(0)

    test_edge_index = torch.tensor(test_edge_index, dtype=torch.long).t().contiguous()
    test_edge_label = torch.tensor(test_edge_label, dtype=torch.float)

    data = Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
    )

    return data, test_edge_index, test_edge_label


def _run_walk_embeddings_cross(
    pair: TaskPair,
    out_dir: Path,
    seed: int,
    task: str,
    walk_len: int,
    walks_per_node: int,
    window: int,
    dim: int,
    p: float,
    q: float,
    epochs: int,
    early_stop_patience: int,
    early_stop_min_delta: float,
    node2vec_cache_dir: Path,
    neg_multiplier: int = 20,
) -> Dict:
    recall_grid = build_recall_grid(step=0.01, end=0.1)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_all_seeds(seed)

    train_df = prepare_dataset(pair.train_path, weight_col="weight")
    test_df = prepare_dataset(pair.test_path, weight_col="weight")

    total_nodes = sorted(
        set(train_df["node1"]).union(set(train_df["node2"]))
        .union(set(test_df["node1"]))
        .union(set(test_df["node2"]))
    )
    train_edges = [tuple(edge) for edge in train_df[["node1", "node2"]].itertuples(index=False, name=None)]
    train_edges_set = {tuple(sorted(edge)) for edge in train_edges}
    test_edges_all = {tuple(sorted(edge)) for edge in test_df[["node1", "node2"]].itertuples(index=False, name=None)}
    pos_edges = sorted(test_edges_all - train_edges_set)

    G_train = nx.Graph()
    for _, row in train_df.iterrows():
        G_train.add_edge(row["node1"], row["node2"], weight=float(row["weight"]))
    G_train.add_nodes_from(total_nodes)

    dataset_payload = {
        "task": task,
        "train_dataset": str(pair.train_path),
        "test_dataset": str(pair.test_path),
        "neg_multiplier": int(neg_multiplier),
        "trials": 1,
        "recall_step": 0.01,
        "recall_end": 0.1,
        "recall_grid": recall_grid,
        "methods": {},
    }

    methods = ["deepwalk", "node2vec"]
    buffers: Dict[str, Dict[str, List]] = {name: {"metrics": [], "ranking": []} for name in methods}

    if not pos_edges:
        empty_metrics = compute_metrics([], [], set(), recall_grid, total_nodes=total_nodes)
        for payload in buffers.values():
            payload["metrics"].append(empty_metrics)
        for method_name, payload in buffers.items():
            stats = aggregate_metrics(payload["metrics"])
            stats["ranking_by_source"] = {}
            dataset_payload["methods"][method_name] = stats
        return dataset_payload

    desired_neg = len(pos_edges) * int(neg_multiplier)
    test_nodes = sorted({n for edge in pos_edges for n in edge})
    neg_edges = _sample_distance2_negatives(
        G_train,
        test_nodes,
        test_edges_all,
        desired_neg,
        seed=seed + 991,
    )
    candidate_pairs = pos_edges + neg_edges
    labels = [1] * len(pos_edges) + [0] * len(neg_edges)
    test_edge_set = set(pos_edges)

    for method in methods:
        cache_path = None
        if method == "node2vec":
            cache_path = _node2vec_cache_path(node2vec_cache_dir, pair, seed, task)
        embeddings = _train_embeddings(
            G_train,
            walk_len=walk_len,
            walks_per_node=walks_per_node,
            window=window,
            dim=dim,
            seed=seed,
            method=method,
            p=p,
            q=q,
            epochs=epochs,
            early_stop_patience=early_stop_patience,
            early_stop_min_delta=early_stop_min_delta,
            log_prefix=f"{pair.label} {method} task{task}",
            candidate_pairs=candidate_pairs,
            labels=labels,
            test_edge_set=test_edge_set,
            recall_grid=recall_grid,
            num_pos=len(pos_edges),
            node_order=total_nodes,
            cache_path=cache_path,
        )
        eval_result = _evaluate_embeddings(
            embeddings,
            candidate_pairs=candidate_pairs,
            labels=labels,
            test_edge_set=test_edge_set,
            recall_grid=recall_grid,
            num_pos=len(pos_edges),
            total_nodes=total_nodes,
        )
        bucket = buffers[method]
        bucket["metrics"].append(eval_result["metrics"])
        bucket.setdefault("ranking", []).append(eval_result.get("ranking", {}))

    for method_name, payload in buffers.items():
        ranking_payloads = payload.get("ranking", [])
        if ranking_payloads:
            ranking_summary = {}
            for key in ranking_payloads[0].keys():
                if key == "k_list":
                    ranking_summary[key] = ranking_payloads[0][key]
                    continue
                values = [r.get(key) for r in ranking_payloads]
                ranking_summary[key] = float(np.nanmean(values)) if values else float("nan")
        else:
            ranking_summary = {}
        stats = aggregate_metrics(payload["metrics"])
        stats["ranking_by_source"] = ranking_summary
        dataset_payload["methods"][method_name] = stats

    return dataset_payload


def _run_gnn_cross_task(
    pair: TaskPair,
    model_type: str,
    seed: int,
    epochs: int,
    hidden_dim: int,
    embed_dim: int,
    lr: float,
    node2vec_dim: int,
    node2vec_walk_len: int,
    node2vec_walks_per_node: int,
    node2vec_window: int,
    node2vec_p: float,
    node2vec_q: float,
    node2vec_epochs: int,
    early_stop_patience: int,
    early_stop_min_delta: float,
    val_size: float,
    node2vec_cache_dir: Path,
    task: str,
) -> Dict:
    set_all_seeds(seed)
    torch.manual_seed(seed)

    train_df = prepare_dataset(pair.train_path, weight_col="weight")
    test_df = prepare_dataset(pair.test_path, weight_col="weight")

    nodes = sorted(
        set(train_df["node1"]).union(set(train_df["node2"]))
        .union(set(test_df["node1"]))
        .union(set(test_df["node2"]))
    )

    G_train = nx.Graph()
    for _, row in train_df.iterrows():
        G_train.add_edge(row["node1"], row["node2"], weight=float(row["weight"]))
    G_train.add_nodes_from(nodes)

    train_edges = [tuple(edge) for edge in train_df[["node1", "node2"]].itertuples(index=False, name=None)]
    train_edge_set = {tuple(sorted(edge)) for edge in train_edges}
    test_edges_all = {tuple(sorted(edge)) for edge in test_df[["node1", "node2"]].itertuples(index=False, name=None)}
    test_edges = sorted(test_edges_all - train_edge_set)

    if not test_edges:
        return {
            "auc": float("nan"),
            "ap": float("nan"),
            "precision@recall_mean": float("nan"),
            "train_time": 0.0,
            "eval_time": 0.0,
        }

    val_edges = []
    if early_stop_patience > 0 and 0.0 < val_size < 0.5:
        train_edges, val_edges = train_test_split(
            train_edges, test_size=val_size, random_state=seed
        )

    cache_path = _node2vec_cache_path(node2vec_cache_dir, pair, seed, task)
    node_features = _load_node2vec_cache(cache_path, nodes)
    if node_features is None:
        node_features = build_node2vec_features(
            G_train,
            nodes,
            dim=node2vec_dim,
            walk_len=node2vec_walk_len,
            walks_per_node=node2vec_walks_per_node,
            window=node2vec_window,
            p=node2vec_p,
            q=node2vec_q,
            seed=seed,
            epochs=node2vec_epochs,
            log_prefix=f"{pair.label} task{task}",
        )

    data, test_edge_index, test_edge_label = _prepare_pyg_data_cross(
        G_train,
        train_edges,
        test_edges,
        nodes,
        node_features,
        test_neg_multiplier=20,
        banned_edge_set=test_edges_all,
    )

    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    train_pos_edges = []
    for u, v in train_edges:
        if u in node_to_idx and v in node_to_idx:
            train_pos_edges.append([node_to_idx[u], node_to_idx[v]])
    train_edge_index = torch.tensor(train_pos_edges, dtype=torch.long).t().contiguous()

    num_train_pos = len(train_pos_edges)
    train_neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=len(nodes),
        num_neg_samples=num_train_pos,
    )

    train_edge_index = torch.cat([train_edge_index, train_neg_edge_index], dim=1)
    train_edge_label = torch.cat([
        torch.ones(num_train_pos),
        torch.zeros(num_train_pos),
    ])

    val_edge_index = None
    val_edge_label = None
    if val_edges:
        val_pos_edges = []
        for u, v in val_edges:
            if u in node_to_idx and v in node_to_idx:
                val_pos_edges.append([node_to_idx[u], node_to_idx[v]])
        val_edge_index = torch.tensor(val_pos_edges, dtype=torch.long).t().contiguous()
        num_val_pos = len(val_pos_edges)
        desired_val_neg = num_val_pos * 20
        val_nodes = sorted({n for edge in val_edges for n in edge})
        banned_val_edges = {tuple(sorted(e)) for e in val_edges}
        val_neg_pairs = _sample_distance2_negatives(
            G_train,
            val_nodes,
            banned_val_edges,
            desired_val_neg,
            seed=seed + 1991,
        )
        val_neg_edge_index = torch.tensor(
            [[node_to_idx[u], node_to_idx[v]] for u, v in val_neg_pairs if u in node_to_idx and v in node_to_idx],
            dtype=torch.long,
        ).t().contiguous()
        val_edge_index = torch.cat([val_edge_index, val_neg_edge_index], dim=1)
        val_edge_label = torch.cat([
            torch.ones(num_val_pos),
            torch.zeros(val_neg_edge_index.size(1)),
        ])

    in_channels = data.x.size(1)
    if model_type == "GCN":
        model = GCN(in_channels, hidden_dim, embed_dim)
    elif model_type == "GraphSAGE":
        model = GraphSAGE(in_channels, hidden_dim, embed_dim)
    elif model_type == "GAT":
        model = GAT(in_channels, hidden_dim, embed_dim, heads=4)
    elif model_type == "GraphConv":
        model = GraphConvNet(in_channels, hidden_dim, embed_dim)
    elif model_type == "GIN":
        model = GIN(in_channels, hidden_dim, embed_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    device = "cuda" if torch.cuda.is_available() else "cpu"

    start_time = time.time()
    model = train_gnn(
        model,
        data,
        train_edge_index,
        train_edge_label,
        epochs=epochs,
        lr=lr,
        device=device,
        val_edge_index=val_edge_index,
        val_edge_label=val_edge_label,
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
    )
    train_time = time.time() - start_time

    start_time = time.time()
    metrics = evaluate_gnn(
        model,
        data,
        test_edge_index,
        test_edge_label,
        test_edge_set={tuple(sorted(edge)) for edge in test_edges},
        node_to_idx=node_to_idx,
        nodes=nodes,
        recall_grid=build_recall_grid(step=0.01, end=0.1),
        device=device,
    )
    eval_time = time.time() - start_time

    metrics["auc"] = metrics.get("auroc")
    metrics["ap"] = metrics.get("auprc")
    metrics["precision@recall_mean"] = metrics.get("precision_at_recall_mean")
    metrics["train_time"] = train_time
    metrics["eval_time"] = eval_time
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Run learning baselines (walk embeddings + GNNs).")
    parser.add_argument("--walk-out-dir", type=Path, default=Path("reports"))
    parser.add_argument("--gnn-out-dir", type=Path, default=Path("reports"))
    parser.add_argument("--models", nargs="+", default=["GCN", "GraphSAGE", "GAT", "GraphConv", "GIN"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--task", type=str, default=TASK_WITHIN, choices=[TASK_WITHIN, TASK_CROSS_PERIOD])
    parser.add_argument("--cross-windows", nargs="*", type=int, default=[30, 90, 180], help="Long windows for task B.")

    parser.add_argument("--gnn-epochs", type=int, default=200)
    parser.add_argument("--gnn-early-stop-patience", type=int, default=5)
    parser.add_argument("--gnn-early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--gnn-hidden-dim", type=int, default=64)
    parser.add_argument("--gnn-embed-dim", type=int, default=32)
    parser.add_argument("--gnn-lr", type=float, default=0.01)
    parser.add_argument("--gnn-val-size", type=float, default=0.1)

    parser.add_argument("--node2vec-dim", type=int, default=64)
    parser.add_argument("--node2vec-walk-len", type=int, default=40)
    parser.add_argument("--node2vec-walks-per-node", type=int, default=10)
    parser.add_argument("--node2vec-window", type=int, default=5)
    parser.add_argument("--node2vec-p", type=float, default=1.0)
    parser.add_argument("--node2vec-q", type=float, default=1.0)
    parser.add_argument("--node2vec-epochs", type=int, default=40)
    parser.add_argument("--node2vec-cache-dir", type=Path, default=Path("reports/node2vec_cache"))

    parser.add_argument("--walk-epochs", type=int, default=40)
    parser.add_argument("--walk-early-stop-patience", type=int, default=5)
    parser.add_argument("--walk-early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--walk-len", type=int, default=40)
    parser.add_argument("--walks-per-node", type=int, default=10)
    parser.add_argument("--walk-window", type=int, default=5)
    parser.add_argument("--walk-dim", type=int, default=64)
    parser.add_argument("--walk-p", type=float, default=1.0)
    parser.add_argument("--walk-q", type=float, default=1.0)
    args = parser.parse_args()

    task_norm = args.task.upper()
    if task_norm not in {TASK_WITHIN, TASK_CROSS_PERIOD}:
        raise ValueError(f"Unknown task '{args.task}' (expected A or B)")

    args.walk_out_dir.mkdir(parents=True, exist_ok=True)
    args.gnn_out_dir.mkdir(parents=True, exist_ok=True)
    args.node2vec_cache_dir.mkdir(parents=True, exist_ok=True)

    if task_norm == TASK_WITHIN:
        datasets = resolve_dataset_paths(MAX_EDGE_ROWS)
        print(f"[Learning] Datasets (<= {MAX_EDGE_ROWS} edges): {len(datasets)}")

        print("[Learning] Checking walk embedding outputs...")
        walk_test_size = 0.5
        walk_datasets = []
        for csv_path in datasets:
            walk_json = args.walk_out_dir / f"{csv_path.stem}_walk_embeddings.json"
            node2vec_cache = (
                args.node2vec_cache_dir
                / f"{csv_path.stem}_seed{args.seed}_test{walk_test_size:.2f}_node2vec_walk.npz"
            )
            if walk_json.is_file() or node2vec_cache.is_file():
                print(f"[Learning] Skip walk embeddings for {csv_path.name} (cached)")
                continue
            walk_datasets.append(csv_path)

        if walk_datasets:
            print("[Learning] Running walk embeddings...")
            run_walk_embeddings(
                out_dir=args.walk_out_dir,
                trials=1,
                neg_multiplier=20,
                test_size=walk_test_size,
                walk_len=args.walk_len,
                walks_per_node=args.walks_per_node,
                window=args.walk_window,
                dim=args.walk_dim,
                p=args.walk_p,
                q=args.walk_q,
                epochs=args.walk_epochs,
                early_stop_patience=args.walk_early_stop_patience,
                early_stop_min_delta=args.walk_early_stop_min_delta,
                dataset_paths=walk_datasets,
                node2vec_cache_dir=args.node2vec_cache_dir,
            )
        else:
            print("[Learning] Walk embeddings already cached. Skipping.")

        print("[Learning] Running GNN baselines...")
        from GNN import run_gnn_experiment

        all_results = {}
        total_tasks = len(args.models) * len(datasets)
        progress = tqdm(total=total_tasks, desc="GNN overall", unit="dataset")
        for model_type in args.models:
            print(f"\n{'='*60}")
            print(f"Running {model_type}")
            print(f"{'='*60}")
            model_results = {}
            for csv_path in datasets:
                if not csv_path.exists():
                    print(f"Skipping {csv_path} (not found)")
                    progress.update(1)
                    continue

                print(f"\nDataset: {csv_path.name}")
                try:
                    metrics = run_gnn_experiment(
                        csv_path=csv_path,
                        weight_col="benecount",
                        model_type=model_type,
                        test_size=0.5,
                        epochs=args.gnn_epochs,
                        hidden_dim=args.gnn_hidden_dim,
                        embed_dim=args.gnn_embed_dim,
                        lr=args.gnn_lr,
                        seed=args.seed,
                        use_node2vec=True,
                        node2vec_dim=args.node2vec_dim,
                        node2vec_walk_len=args.node2vec_walk_len,
                        node2vec_walks_per_node=args.node2vec_walks_per_node,
                        node2vec_window=args.node2vec_window,
                        node2vec_p=args.node2vec_p,
                        node2vec_q=args.node2vec_q,
                        node2vec_epochs=args.node2vec_epochs,
                        node2vec_cache_dir=args.node2vec_cache_dir,
                        early_stop_patience=args.gnn_early_stop_patience,
                        early_stop_min_delta=args.gnn_early_stop_min_delta,
                        val_size=args.gnn_val_size,
                    )
                    print(f"  AUC: {metrics['auc']:.4f}")
                    print(f"  AP: {metrics['ap']:.4f}")
                    print(f"  Precision@Recall: {metrics['precision@recall_mean']:.4f}")
                    print(f"  Train time: {metrics['train_time']:.2f}s")
                    print(f"  Eval time: {metrics['eval_time']:.2f}s")
                    model_results[csv_path.name] = metrics
                except Exception as exc:
                    print(f"  Error: {exc}")
                    model_results[csv_path.name] = {"error": str(exc)}
                progress.update(1)
            all_results[model_type] = model_results
        progress.close()

        output_file = args.gnn_out_dir / "gnn_node2vec_results.json"
        output_file.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
        print(f"\nResults saved to {output_file}")
        return

    pairs = _resolve_task_b_pairs(MAX_EDGE_ROWS, args.cross_windows)

    print(f"[Learning] Task {task_norm} pairs: {len(pairs)}")
    if not pairs:
        print("[Learning] No eligible dataset pairs found.")
        return

    print("[Learning] Checking walk embedding outputs...")
    walk_pairs = []
    for pair in pairs:
        walk_json = args.walk_out_dir / f"{pair.label}_walk_embeddings_task_{task_norm}.json"
        node2vec_cache = _node2vec_cache_path(args.node2vec_cache_dir, pair, args.seed, task_norm)
        if walk_json.is_file() or node2vec_cache.is_file():
            print(f"[Learning] Skip walk embeddings for {pair.label} (cached)")
            continue
        walk_pairs.append(pair)

    if walk_pairs:
        print("[Learning] Running walk embeddings...")
        for pair in walk_pairs:
            payload = _run_walk_embeddings_cross(
                pair=pair,
                out_dir=args.walk_out_dir,
                seed=args.seed,
                task=task_norm,
                walk_len=args.walk_len,
                walks_per_node=args.walks_per_node,
                window=args.walk_window,
                dim=args.walk_dim,
                p=args.walk_p,
                q=args.walk_q,
                epochs=args.walk_epochs,
                early_stop_patience=args.walk_early_stop_patience,
                early_stop_min_delta=args.walk_early_stop_min_delta,
                node2vec_cache_dir=args.node2vec_cache_dir,
            )
            out_path = args.walk_out_dir / f"{pair.label}_walk_embeddings_task_{task_norm}.json"
            out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            print(f"[Done] {pair.label} -> {out_path}")
    else:
        print("[Learning] Walk embeddings already cached. Skipping.")

    print("[Learning] Running GNN baselines...")
    all_results = {}
    total_tasks = len(args.models) * len(pairs)
    progress = tqdm(total=total_tasks, desc="GNN overall", unit="dataset")
    for model_type in args.models:
        print(f"\n{'='*60}")
        print(f"Running {model_type}")
        print(f"{'='*60}")
        model_results = {}
        for pair in pairs:
            print(f"\nDataset: {pair.label}")
            try:
                metrics = _run_gnn_cross_task(
                    pair=pair,
                    model_type=model_type,
                    seed=args.seed,
                    epochs=args.gnn_epochs,
                    hidden_dim=args.gnn_hidden_dim,
                    embed_dim=args.gnn_embed_dim,
                    lr=args.gnn_lr,
                    node2vec_dim=args.node2vec_dim,
                    node2vec_walk_len=args.node2vec_walk_len,
                    node2vec_walks_per_node=args.node2vec_walks_per_node,
                    node2vec_window=args.node2vec_window,
                    node2vec_p=args.node2vec_p,
                    node2vec_q=args.node2vec_q,
                    node2vec_epochs=args.node2vec_epochs,
                    early_stop_patience=args.gnn_early_stop_patience,
                    early_stop_min_delta=args.gnn_early_stop_min_delta,
                    val_size=args.gnn_val_size,
                    node2vec_cache_dir=args.node2vec_cache_dir,
                    task=task_norm,
                )
                print(f"  AUC: {metrics.get('auc', float('nan')):.4f}")
                print(f"  AP: {metrics.get('ap', float('nan')):.4f}")
                print(f"  Precision@Recall: {metrics.get('precision@recall_mean', float('nan')):.4f}")
                print(f"  Train time: {metrics.get('train_time', 0.0):.2f}s")
                print(f"  Eval time: {metrics.get('eval_time', 0.0):.2f}s")
                model_results[pair.label] = metrics
            except Exception as exc:
                print(f"  Error: {exc}")
                model_results[pair.label] = {"error": str(exc)}
            progress.update(1)
        all_results[model_type] = model_results
    progress.close()

    output_file = args.gnn_out_dir / f"gnn_node2vec_results_task_{task_norm}.json"
    output_file.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"\nResults saved to {output_file}")


if __name__ == "__main__":
    main()

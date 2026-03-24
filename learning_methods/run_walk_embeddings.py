from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import networkx as nx
import numpy as np
import sys
from pathlib import Path as PathLib

# Add parent directory to path for imports
sys.path.insert(0, str(PathLib(__file__).parent.parent))

from h3.h3_core import (
    GraphCache,
    prepare_dataset,
    set_all_seeds,
    split_train_test,
)

from utils.metrics import build_recall_grid, compute_metrics, aggregate_metrics, ranking_metrics_by_source


DATASET_ROOTS = [
    Path("data/2014data"),
    Path("data/2015data"),
]
DATASET_PATTERN = "*_30.csv"
MAX_EDGE_ROWS = 300_000
NEG_MULTIPLIER = 20
TEST_SIZE = 0.5
TRIALS = 1
RANKING_K = [50, 100, 500]


def _edge_set_from_df(df) -> List[Tuple]:
    edges = set()
    for u, v in df[["node1", "node2"]].itertuples(index=False, name=None):
        if u == v:
            continue
        pair = (u, v) if u < v else (v, u)
        edges.add(pair)
    return sorted(edges)


def _count_edges(csv_path: Path) -> int:
    with csv_path.open("r", encoding="utf-8") as f:
        next(f, None)
        return sum(1 for _ in f)


def resolve_dataset_paths(max_edge_rows: int = MAX_EDGE_ROWS) -> List[Path]:
    paths: List[Path] = []
    for root in DATASET_ROOTS:
        if not root.is_dir():
            continue
        for path in sorted(root.glob(DATASET_PATTERN)):
            if not path.is_file():
                continue
            edge_count = _count_edges(path)
            if edge_count > max_edge_rows:
                continue
            paths.append(path)
    return paths


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










def _random_walk(graph: nx.Graph, start, walk_len: int, rng: random.Random) -> List[str]:
    walk = [start]
    while len(walk) < walk_len:
        cur = walk[-1]
        neighbors = list(graph.neighbors(cur))
        if not neighbors:
            break
        walk.append(rng.choice(neighbors))
    return [str(n) for n in walk]


def _node2vec_walk(
    graph: nx.Graph,
    start,
    walk_len: int,
    p: float,
    q: float,
    rng: random.Random,
) -> List[str]:
    walk = [start]
    while len(walk) < walk_len:
        cur = walk[-1]
        neighbors = list(graph.neighbors(cur))
        if not neighbors:
            break
        if len(walk) == 1:
            walk.append(rng.choice(neighbors))
            continue
        prev = walk[-2]
        probs = []
        for nxt in neighbors:
            if nxt == prev:
                weight = 1.0 / p
            elif graph.has_edge(nxt, prev):
                weight = 1.0
            else:
                weight = 1.0 / q
            probs.append(weight)
        total = sum(probs)
        r = rng.random() * total
        acc = 0.0
        for nxt, w in zip(neighbors, probs):
            acc += w
            if acc >= r:
                walk.append(nxt)
                break
    return [str(n) for n in walk]


def _train_embeddings(
    graph: nx.Graph,
    walk_len: int,
    walks_per_node: int,
    window: int,
    dim: int,
    seed: int,
    method: str,
    p: float,
    q: float,
    epochs: int,
    early_stop_patience: int,
    early_stop_min_delta: float,
    log_prefix: str,
    candidate_pairs: Sequence[Tuple],
    labels: List[int],
    test_edge_set: set[Tuple],
    recall_grid: List[float],
    num_pos: int,
    node_order: List,
    cache_path: Path | None,
):
    try:
        from gensim.models import Word2Vec
    except Exception as exc:
        raise RuntimeError("gensim is required for DeepWalk/node2vec embeddings") from exc

    rng = random.Random(seed)
    nodes = list(graph.nodes())
    walks: List[List[str]] = []
    for _ in range(walks_per_node):
        rng.shuffle(nodes)
        for node in nodes:
            if method == "deepwalk":
                walk = _random_walk(graph, node, walk_len, rng)
            else:
                walk = _node2vec_walk(graph, node, walk_len, p, q, rng)
            walks.append(walk)

    model = Word2Vec(
        vector_size=dim,
        window=window,
        min_count=0,
        sg=1,
        workers=os.cpu_count() or 1,
        seed=seed,
        compute_loss=True,
    )
    model.build_vocab(walks)
    best_pr_auc = -float("inf")
    best_embeddings: np.ndarray | None = None
    patience = 0
    prev_cum_loss = 0.0
    for epoch in range(1, max(1, epochs) + 1):
        model.train(walks, total_examples=len(walks), epochs=1, compute_loss=True)
        cum_loss = model.get_latest_training_loss()
        epoch_loss = cum_loss - prev_cum_loss
        prev_cum_loss = cum_loss

        eval_result = _evaluate_embeddings(
            model.wv,
            candidate_pairs=candidate_pairs,
            labels=labels,
            test_edge_set=test_edge_set,
            recall_grid=recall_grid,
            num_pos=num_pos,
            total_nodes=node_order,
        )
        print(
            f"[{log_prefix}] epoch {epoch}/{epochs} "
            f"loss={epoch_loss:.6f} auprc={eval_result['metrics']['auprc']:.6f} "
            f"auroc={eval_result['metrics']['auroc']:.6f} "
            f"precision_mean={eval_result['metrics']['precision_at_recall_mean']:.6f} "
            f"best_pr_auc={best_pr_auc:.6f} patience={patience}/{early_stop_patience}"
        )

        if eval_result["metrics"]["auprc"] - best_pr_auc > early_stop_min_delta:
            best_pr_auc = eval_result["metrics"]["auprc"]
            patience = 0
            if cache_path is not None and method == "node2vec":
                best_embeddings = np.vstack([model.wv[str(n)] for n in node_order])
        else:
            patience += 1
            if early_stop_patience > 0 and patience >= early_stop_patience:
                print(f"[{log_prefix}] early stop at epoch {epoch}")
                break
    if cache_path is not None and method == "node2vec" and best_embeddings is not None:
        np.savez(cache_path, nodes=np.array(node_order), embeddings=best_embeddings)
        print(f"[{log_prefix}] saved node2vec cache to {cache_path}")
    return model.wv


def _score_pairs(embeddings, pairs: Sequence[Tuple]) -> List[float]:
    scores = []
    for u, v in pairs:
        if str(u) not in embeddings or str(v) not in embeddings:
            scores.append(float("nan"))
            continue
        vec_u = embeddings[str(u)]
        vec_v = embeddings[str(v)]
        denom = float(np.linalg.norm(vec_u) * np.linalg.norm(vec_v))
        if denom <= 0.0:
            scores.append(float("nan"))
        else:
            scores.append(float(np.dot(vec_u, vec_v) / denom))
    return scores


def _evaluate_embeddings(
    embeddings,
    candidate_pairs: Sequence[Tuple],
    labels: List[int],
    test_edge_set: set[Tuple],
    recall_grid: List[float],
    num_pos: int,
    total_nodes: Sequence,
) -> Dict:
    scores = _score_pairs(embeddings, candidate_pairs)
    metrics = compute_metrics(
        candidate_pairs,
        scores,
        test_edge_set,
        recall_grid,
        total_nodes=total_nodes,
    )
    ranking = ranking_metrics_by_source(
        candidate_pairs,
        labels,
        scores,
        k_list=metrics.get("k_list", []),
    )
    return {
        "scores": scores,
        "metrics": metrics,
        "ranking": ranking,
    }




def run(
    out_dir: Path,
    trials: int,
    neg_multiplier: int,
    test_size: float,
    walk_len: int,
    walks_per_node: int,
    window: int,
    dim: int,
    p: float,
    q: float,
    epochs: int,
    early_stop_patience: int,
    early_stop_min_delta: float,
    dataset_paths: Sequence[Path] | None = None,
    node2vec_cache_dir: Path | None = None,
) -> Dict:
    recall_grid = build_recall_grid(step=0.01, end=0.1)
    out_dir.mkdir(parents=True, exist_ok=True)
    all_outputs = {}

    datasets = list(dataset_paths) if dataset_paths else resolve_dataset_paths()
    for dataset_path in datasets:
        if not dataset_path.is_file():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        data = prepare_dataset(dataset_path, weight_col="weight")
        total_nodes = sorted(set(data["node1"]).union(set(data["node2"])))
        all_edges = set(_edge_set_from_df(data))

        dataset_payload = {
            "dataset": str(dataset_path),
            "neg_multiplier": int(neg_multiplier),
            "trials": trials,
            "test_size": test_size,
            "recall_step": 0.01,
            "recall_end": 0.1,
            "recall_grid": recall_grid,
            "methods": {},
        }

        methods = ["deepwalk", "node2vec"]
        buffers: Dict[str, Dict[str, List]] = {}
        for name in methods:
            buffers[name] = {"metrics": [], "ranking": []}

        for trial_idx in range(trials):
            seed = 42 + trial_idx
            set_all_seeds(seed)

            X_train, X_test = split_train_test(data, test_size=test_size, seed=seed)
            G_train = nx.Graph()
            for _, row in X_train.iterrows():
                G_train.add_edge(row["node1"], row["node2"], weight=float(row["weight"]))
            G_train.add_nodes_from(total_nodes)

            pos_edges = [tuple(sorted(edge)) for edge in X_test[["node1", "node2"]].itertuples(index=False, name=None)]
            pos_edges = sorted(set(pos_edges))
            if not pos_edges:
                for payload in buffers.values():
                    payload["metrics"].append({
                        "auroc": float("nan"),
                        "auprc": float("nan"),
                        "precision_at_recall_mean": float("nan"),
                        "precision_curve": [float("nan")] * len(recall_grid),
                        "precision_at_k": {},
                        "recall_at_k": {},
                        "ndcg_at_k": {},
                        "coverage_at_k": {},
                        "k_list": [],
                        "recall_grid": list(recall_grid),
                    })
                continue

            desired_neg = len(pos_edges) * int(neg_multiplier)
            test_nodes = sorted(set(X_test["node1"]).union(set(X_test["node2"])))
            neg_edges = _sample_distance2_negatives(
                G_train,
                test_nodes,
                all_edges,
                desired_neg,
                seed=seed + 991,
            )
            candidate_pairs = pos_edges + neg_edges
            labels = [1] * len(pos_edges) + [0] * len(neg_edges)
            test_edge_set = set(pos_edges)

            for method in methods:
                cache_path = None
                if method == "node2vec" and node2vec_cache_dir is not None:
                    node2vec_cache_dir.mkdir(parents=True, exist_ok=True)
                    cache_name = (
                        f"{dataset_path.stem}_seed{seed}_test{test_size:.2f}"
                        f"_node2vec_walk.npz"
                    )
                    cache_path = node2vec_cache_dir / cache_name
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
                    log_prefix=f"{dataset_path.stem} {method} trial {trial_idx + 1}",
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

        out_path = out_dir / f"{dataset_path.stem}_walk_embeddings.json"
        out_path.write_text(json.dumps(dataset_payload, indent=2), encoding="utf-8")
        print(f"[Done] {dataset_path.name} -> {out_path}")
        all_outputs[str(dataset_path)] = dataset_payload

    return all_outputs


def main() -> None:
    parser = argparse.ArgumentParser(description="DeepWalk/node2vec baselines on fixed datasets.")
    parser.add_argument("--out-dir", type=Path, default=Path("reports/walk_embeddings"))
    parser.add_argument("--trials", type=int, default=TRIALS)
    parser.add_argument("--neg-multiplier", type=int, default=NEG_MULTIPLIER)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--walk-len", type=int, default=40)
    parser.add_argument("--walks-per-node", type=int, default=10)
    parser.add_argument("--window", type=int, default=5)
    parser.add_argument("--dim", type=int, default=64)
    parser.add_argument("--p", type=float, default=1.0)
    parser.add_argument("--q", type=float, default=1.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--early-stop-patience", type=int, default=20)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    args = parser.parse_args()

    run(
        out_dir=args.out_dir,
        trials=args.trials,
        neg_multiplier=args.neg_multiplier,
        test_size=args.test_size,
        walk_len=args.walk_len,
        walks_per_node=args.walks_per_node,
        window=args.window,
        dim=args.dim,
        p=args.p,
        q=args.q,
        epochs=args.epochs,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
    )


if __name__ == "__main__":
    main()

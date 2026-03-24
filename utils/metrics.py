from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


# ---------------------------------------------------------------------------
# Recall grid
# ---------------------------------------------------------------------------

def build_recall_grid(step: float = 0.01, end: float = 0.1) -> List[float]:
    step = max(step, 1e-6)
    end = max(end, step)
    num_points = int(round(end / step))
    grid = [0.0]
    for i in range(1, num_points + 1):
        grid.append(round(i * step, 6))
    return grid


# ---------------------------------------------------------------------------
# AUROC / AUPRC  (kept as reference baselines)
# ---------------------------------------------------------------------------

def _safe_roc_auc(labels: List[int], scores: List[float]) -> float:
    if not labels or len(set(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def _safe_pr_auc(labels: List[int], scores: List[float]) -> float:
    if not labels or len(set(labels)) < 2:
        return float("nan")
    return float(average_precision_score(labels, scores))


def compute_auroc_auprc(
    labels: Sequence[int], scores: Sequence[float]
) -> Tuple[float, float]:
    return _safe_roc_auc(list(labels), list(scores)), _safe_pr_auc(
        list(labels), list(scores)
    )


# ---------------------------------------------------------------------------
# nDCG helpers
# ---------------------------------------------------------------------------

def _dcg_at_k(labels_ranked: List[int], k: int) -> float:
    dcg = 0.0
    for i, rel in enumerate(labels_ranked[:k]):
        if rel:
            dcg += 1.0 / math.log2(i + 2)
    return dcg


def _ndcg_at_k(labels_ranked: List[int], k: int, num_pos: int) -> float:
    if k <= 0 or num_pos <= 0:
        return 0.0
    dcg = _dcg_at_k(labels_ranked, k)
    ideal_k = min(k, num_pos)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_k))
    return float(dcg / idcg) if idcg > 0 else 0.0


# ---------------------------------------------------------------------------
# Precision@Recall%  (bug-fixed: max_k now based on candidate length)
# ---------------------------------------------------------------------------

def precision_at_recall(
    rank_pairs: Sequence[Tuple],
    pos_edge_set: set,
    num_pos: int,
    recall_grid: Sequence[float],
) -> List[float]:
    """
    For each recall level r in recall_grid, return the precision achieved
    at the smallest candidate list that covers r * num_pos positive edges.

    Bug fix vs. original: cumulative list now spans the full candidate set
    (up to the index needed to reach max_recall), rather than being capped
    at ceil(max_recall * num_pos) which underestimates list length when
    positives are sparse (high negative ratio).
    """
    if not rank_pairs or num_pos <= 0:
        return [float("nan")] * len(recall_grid)

    max_recall = recall_grid[-1] if recall_grid else 0.0
    if max_recall <= 0:
        return [0.0] * len(recall_grid)

    target_hits = int(math.ceil(max_recall * num_pos))  # how many pos we need
    cumulative: List[float] = []
    hits = 0

    # Traverse the full ranked list until we have enough positive hits
    for idx, pair in enumerate(rank_pairs):
        if tuple(sorted(pair)) in pos_edge_set:
            hits += 1
        cumulative.append(hits / (idx + 1))
        if hits >= target_hits:
            break

    precisions: List[float] = []
    for recall in recall_grid:
        if recall <= 0:
            precisions.append(0.0)
            continue
        needed_hits = int(math.ceil(recall * num_pos))
        # Find first index where cumulative hits reach needed_hits
        found = False
        running = 0
        for idx, pair in enumerate(rank_pairs):
            if tuple(sorted(pair)) in pos_edge_set:
                running += 1
            if running >= needed_hits:
                prec = running / (idx + 1)
                precisions.append(float(prec))
                found = True
                break
        if not found:
            precisions.append(float("nan"))

    return precisions


# ---------------------------------------------------------------------------
# Lift@K
# ---------------------------------------------------------------------------

def lift_at_k(
    labels_ranked: List[int],
    k: int,
    base_rate: float,
) -> float:
    """
    Lift@K = Precision@K / base_rate.
    base_rate = num_pos / total_candidates (computed externally for clarity).
    """
    if k <= 0 or base_rate <= 0:
        return float("nan")
    k_eff = min(k, len(labels_ranked))
    if k_eff == 0:
        return float("nan")
    prec = sum(labels_ranked[:k_eff]) / k_eff
    return float(prec / base_rate)


# ---------------------------------------------------------------------------
# Mean Rank Percentile of positive edges
# ---------------------------------------------------------------------------

def mean_rank_percentile(
    pairs: Sequence[Tuple],
    scores: Sequence[float],
    pos_edge_set: set,
) -> float:
    """
    For every positive edge, record its rank percentile in the scored list
    (rank 1 = top, percentile = rank / total).  Return the mean over all
    positive edges.  Lower is better.
    """
    if not pairs:
        return float("nan")
    ranked = sorted(zip(pairs, scores), key=lambda x: x[1], reverse=True)
    total = len(ranked)
    percentiles: List[float] = []
    for rank_0based, (pair, _score) in enumerate(ranked):
        if tuple(sorted(pair)) in pos_edge_set:
            percentiles.append((rank_0based + 1) / total)
    if not percentiles:
        return float("nan")
    return float(np.mean(percentiles))


# ---------------------------------------------------------------------------
# Default K list
# ---------------------------------------------------------------------------

def default_k_list(
    num_pos: int,
    candidate_len: int,
    max_k: int = 10000,
    base: Sequence[int] = (100, 500, 1000),
) -> List[int]:
    ks = {k for k in base if 0 < k <= min(candidate_len, max_k)}
    if not ks:
        ks = {min(candidate_len, max_k)} if candidate_len > 0 else {1}
    return sorted(ks)


# ---------------------------------------------------------------------------
# Core compute_metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    pairs: Sequence[Tuple],
    scores: Sequence[float],
    pos_edge_set: set,
    recall_grid: Sequence[float],
    k_list: Optional[Sequence[int]] = None,
    total_nodes: Optional[Sequence] = None,
) -> Dict:
    """
    Returns per-trial metrics dict.  Coverage@K and Recall@K are removed.
    New metrics: lift_at_k, mean_rank_percentile, precision_at_recall (fixed).
    AUROC is retained as a reference baseline.

    `total_nodes` is accepted for backward compatibility with older runners
    and is intentionally ignored here.
    """
    nan = float("nan")
    empty = {
        "auroc": nan,
        "auprc": nan,
        "precision_at_recall_mean": nan,
        "precision_curve": [nan] * len(recall_grid),
        "precision_at_k": {},
        "lift_at_k": {},
        "ndcg_at_k": {},
        "mrr": nan,
        "mean_rank_percentile": nan,
        "k_list": [],
        "recall_grid": list(recall_grid),
    }
    if not pairs:
        return empty

    labels = [1 if tuple(sorted(p)) in pos_edge_set else 0 for p in pairs]
    num_pos = int(sum(labels))
    total = len(labels)
    base_rate = num_pos / total if total > 0 else float("nan")

    auroc = _safe_roc_auc(labels, list(scores))
    auprc = _safe_pr_auc(labels, list(scores))

    ranked = sorted(zip(pairs, scores, labels), key=lambda x: x[1], reverse=True)
    rank_pairs = [tuple(sorted(pair)) for pair, _, _ in ranked]
    labels_ranked = [lbl for _, _, lbl in ranked]

    if k_list is None:
        k_list = default_k_list(num_pos, len(rank_pairs))
    k_list = [int(k) for k in k_list if k > 0]

    # --- P@Recall% (bug-fixed) ---
    precisions = precision_at_recall(rank_pairs, pos_edge_set, num_pos, recall_grid)
    precision_mean = float(np.nanmean(precisions)) if precisions else nan

    # --- Mean Rank Percentile ---
    mrp = mean_rank_percentile(pairs, list(scores), pos_edge_set)

    # --- MRR (global, score-ranked) ---
    mrr_val = nan
    for idx, lbl in enumerate(labels_ranked, start=1):
        if lbl == 1:
            mrr_val = 1.0 / idx
            break

    # --- Per-K metrics ---
    precision_at_k: Dict[int, float] = {}
    lift_at_k_dict: Dict[int, float] = {}
    ndcg_at_k: Dict[int, float] = {}

    for k in k_list:
        k_eff = min(k, len(labels_ranked))
        if k_eff <= 0:
            precision_at_k[k] = 0.0
            lift_at_k_dict[k] = nan
            ndcg_at_k[k] = 0.0
            continue
        hits = sum(labels_ranked[:k_eff])
        prec = hits / k_eff
        precision_at_k[k] = float(prec)
        lift_at_k_dict[k] = float(prec / base_rate) if base_rate > 0 else nan
        ndcg_at_k[k] = _ndcg_at_k(labels_ranked, k_eff, num_pos)

    return {
        "auroc": float(auroc),
        "auprc": float(auprc),
        "precision_at_recall_mean": float(precision_mean),
        "precision_curve": list(map(float, precisions)),
        "precision_at_k": precision_at_k,
        "lift_at_k": lift_at_k_dict,
        "ndcg_at_k": ndcg_at_k,
        "mrr": float(mrr_val),
        "mean_rank_percentile": float(mrp),
        "k_list": list(k_list),
        "recall_grid": list(recall_grid),
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def aggregate_curves(curves: List[List[float]]) -> Dict:
    arr = np.array(curves, dtype=np.float64)
    return {
        "trial_means": [float(np.nanmean(c)) for c in arr],
        "overall_mean": float(np.nanmean(arr)),
        "mean_curve": np.nanmean(arr, axis=0).tolist(),
        "min_curve": np.nanmin(arr, axis=0).tolist(),
        "max_curve": np.nanmax(arr, axis=0).tolist(),
    }


def aggregate_metrics(metrics_list: List[Dict]) -> Dict:
    if not metrics_list:
        return {}

    curves = [m.get("precision_curve", []) for m in metrics_list]
    curve_stats = aggregate_curves(curves) if curves else {}

    def mean_of(key: str) -> float:
        vals = [m[key] for m in metrics_list if m.get(key) is not None]
        return float(np.nanmean(vals)) if vals else float("nan")

    def mean_dict(key: str) -> Dict[int, float]:
        buckets: Dict[int, List[float]] = {}
        for m in metrics_list:
            for k, v in m.get(key, {}).items():
                buckets.setdefault(int(k), []).append(float(v))
        return {k: float(np.nanmean(vs)) for k, vs in buckets.items()}

    return {
        "auroc": mean_of("auroc"),
        "auprc": mean_of("auprc"),
        "precision_at_recall_mean": mean_of("precision_at_recall_mean"),
        "precision_at_k": mean_dict("precision_at_k"),
        "lift_at_k": mean_dict("lift_at_k"),
        "ndcg_at_k": mean_dict("ndcg_at_k"),
        "mrr": mean_of("mrr"),
        "mean_rank_percentile": mean_of("mean_rank_percentile"),
        "precision_at_recall": curve_stats,
        "recall_grid": metrics_list[0].get("recall_grid", []),
        "k_list": metrics_list[0].get("k_list", []),
    }


# ---------------------------------------------------------------------------
# Per-source ranking metrics  (MRR + MAP@K; Coverage/Recall@K removed)
# ---------------------------------------------------------------------------

def ranking_metrics_by_source(
    pairs: Sequence[Tuple],
    labels: Sequence[int],
    scores: Sequence[float],
    k_list: Sequence[int],
    base_rate: Optional[float] = None,
) -> Dict:
    """
    Node-level ranking metrics.
    Removed: recall@k, coverage@k.
    Retained: MRR, precision@k, hit@k, map@k, ndcg@k.
    Added: lift@k (requires base_rate).

    MAP@K note: denominator = min(num_pos, k)  — "capped AP", standard in
    recommendation literature (differs from IR-style AP where denom = num_pos).
    Document this choice explicitly when comparing to external baselines.
    """
    per_node: Dict[int, List[Tuple[int, float, int]]] = {}
    for (u, v), score, label in zip(pairs, scores, labels):
        per_node.setdefault(int(u), []).append((int(v), float(score), int(label)))
        per_node.setdefault(int(v), []).append((int(u), float(score), int(label)))

    bucket_keys = ["precision", "hit", "map", "ndcg", "lift"]
    metrics: Dict[int, Dict[str, List[float]]] = {
        k: {bk: [] for bk in bucket_keys} for k in k_list
    }
    mrr_list: List[float] = []
    evaluated_nodes = 0

    for _, items in per_node.items():
        num_pos = sum(lbl for _, _, lbl in items)
        if num_pos == 0:
            continue
        evaluated_nodes += 1
        ranked = sorted(items, key=lambda x: x[1], reverse=True)
        labels_ranked = [lbl for _, _, lbl in ranked]

        # MRR
        first_pos = next(
            (idx for idx, lbl in enumerate(labels_ranked, 1) if lbl == 1), None
        )
        mrr_list.append(1.0 / first_pos if first_pos else 0.0)

        for k in k_list:
            k_eff = min(k, len(labels_ranked))
            hits = sum(labels_ranked[:k_eff])
            prec = hits / k_eff if k_eff > 0 else 0.0

            denom = min(num_pos, k_eff)
            ap_sum = 0.0
            running = 0
            for idx, lbl in enumerate(labels_ranked[:k_eff], 1):
                if lbl == 1:
                    running += 1
                    ap_sum += running / idx
            map_k = ap_sum / denom if denom > 0 else 0.0

            metrics[k]["precision"].append(prec)
            metrics[k]["hit"].append(1.0 if hits > 0 else 0.0)
            metrics[k]["map"].append(map_k)
            metrics[k]["ndcg"].append(_ndcg_at_k(labels_ranked, k_eff, num_pos))
            if base_rate and base_rate > 0:
                metrics[k]["lift"].append(prec / base_rate)

    summary: Dict = {
        "evaluated_nodes": int(evaluated_nodes),
        "mrr": float(np.nanmean(mrr_list)) if mrr_list else float("nan"),
    }
    for k, vals in metrics.items():
        summary[f"precision@{k}"] = float(np.nanmean(vals["precision"])) if vals["precision"] else float("nan")
        summary[f"hit@{k}"] = float(np.nanmean(vals["hit"])) if vals["hit"] else float("nan")
        summary[f"map@{k}"] = float(np.nanmean(vals["map"])) if vals["map"] else float("nan")
        summary[f"ndcg@{k}"] = float(np.nanmean(vals["ndcg"])) if vals["ndcg"] else float("nan")
        if vals["lift"]:
            summary[f"lift@{k}"] = float(np.nanmean(vals["lift"]))
    return summary

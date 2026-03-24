"""
GNN Baseline Experiments for Link Prediction
使用GCN、GraphSAGE、GAT三种方法与H3对比

数据集来自您的h3_hparam_sweep.py中定义的：
- Within-period: 2014/2015各州的30天数据
- Cross-period: 30天→90天、30天→180天
- Cross-year: 2014→2015
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch import nn
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, GraphConv, GINConv
from torch_geometric.utils import negative_sampling
from tqdm import tqdm
import sys
from pathlib import Path as PathLib

# Add parent directory to path for imports
sys.path.insert(0, str(PathLib(__file__).parent.parent))

from utils.metrics import build_recall_grid, compute_metrics, compute_auroc_auprc, ranking_metrics_by_source
from h3.h3_core import prepare_dataset

_TORCH_THREADS_SET = False


def _configure_torch_threads(num_threads: int = 32, num_interop_threads: int = 32) -> None:
    """Set torch thread counts once, before any parallel work starts."""
    global _TORCH_THREADS_SET
    if _TORCH_THREADS_SET:
        return
    try:
        torch.set_num_threads(num_threads)
        torch.set_num_interop_threads(num_interop_threads)
    except RuntimeError:
        # If parallel work already started, skip without failing.
        pass
    _TORCH_THREADS_SET = True


_configure_torch_threads()





class GCN(nn.Module):
    """Graph Convolutional Network"""
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
    
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
    
    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)
    
    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)


class GraphSAGE(nn.Module):
    """GraphSAGE"""
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = SAGEConv(in_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, out_channels)
    
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
    
    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)
    
    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)


class GAT(nn.Module):
    """Graph Attention Network"""
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, heads: int = 4):
        super().__init__()
        self.conv1 = GATConv(in_channels, hidden_channels, heads=heads, concat=True)
        self.conv2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False)
    
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
    
    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)
    
    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)


class GraphConvNet(nn.Module):
    """GraphConv"""
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, out_channels)
    
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
    
    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)
    
    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)


class GIN(nn.Module):
    """Graph Isomorphism Network"""
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        super().__init__()
        mlp1 = nn.Sequential(
            nn.Linear(in_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        mlp2 = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, out_channels),
        )
        self.conv1 = GINConv(mlp1)
        self.conv2 = GINConv(mlp2)
    
    def encode(self, x, edge_index):
        x = self.conv1(x, edge_index).relu()
        x = self.conv2(x, edge_index)
        return x
    
    def decode(self, z, edge_label_index):
        src = z[edge_label_index[0]]
        dst = z[edge_label_index[1]]
        return (src * dst).sum(dim=-1)
    
    def forward(self, x, edge_index, edge_label_index):
        z = self.encode(x, edge_index)
        return self.decode(z, edge_label_index)


# ==================== Data Preparation ====================



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


def prepare_pyg_data(
    G: nx.Graph,
    train_edges: List[Tuple],
    test_edges: List[Tuple],
    use_features: bool = False,
    node_features: Optional[torch.Tensor] = None,
    test_neg_multiplier: int = 20,
) -> Tuple[Data, torch.Tensor, torch.Tensor]:
    """
    将NetworkX图转换为PyTorch Geometric格式
    """
    # 节点映射
    nodes = sorted(G.nodes())
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    
    # 节点特征
    if node_features is not None:
        x = node_features
    elif use_features:
        avg_neighbor_degree = nx.average_neighbor_degree(G)
        features = []
        for node in nodes:
            weighted_degree = G.degree(node, weight="weight")
            unweighted_degree = G.degree(node)
            clustering_coefficient = nx.clustering(G, node)
            neighbor_degree = avg_neighbor_degree.get(node, 0.0)
            features.append([
                weighted_degree,
                unweighted_degree,
                clustering_coefficient,
                neighbor_degree,
            ])
        x = torch.tensor(features, dtype=torch.float)
        if x.size(1) < 32:
            extra = torch.randn(x.size(0), 32 - x.size(1))
            x = torch.cat([x, extra], dim=1)
    else:
        # 使用one-hot编码
        x = torch.eye(len(nodes), dtype=torch.float)
    
    # 训练边（构建消息传递图）
    edge_index = []
    edge_weight = []
    for u, v in train_edges:
        if u not in node_to_idx or v not in node_to_idx:
            continue
        idx_u = node_to_idx[u]
        idx_v = node_to_idx[v]
        # 无向图，添加两个方向
        edge_index.append([idx_u, idx_v])
        edge_index.append([idx_v, idx_u])
        
        w = G[u][v].get('weight', 1.0)
        edge_weight.append(w)
        edge_weight.append(w)
    
    edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
    edge_weight = torch.tensor(edge_weight, dtype=torch.float)
    
    # 测试边标签
    test_edge_index = []
    test_edge_label = []
    
    # Positive samples
    pos_edges = set()
    for u, v in test_edges:
        if u not in node_to_idx or v not in node_to_idx:
            continue
        idx_u = node_to_idx[u]
        idx_v = node_to_idx[v]
        if idx_u != idx_v:
            pair = tuple(sorted([idx_u, idx_v]))
            if pair not in pos_edges:
                pos_edges.add(pair)
                test_edge_index.append([idx_u, idx_v])
                test_edge_label.append(1)
    
    # Negative samples (distance-2, 20x positives)
    num_neg = len(pos_edges) * max(1, int(test_neg_multiplier))
    test_nodes = sorted(set(u for u, v in test_edges).union(set(v for u, v in test_edges)))
    banned = {tuple(sorted(e)) for e in test_edges}
    neg_pairs = _sample_distance2_negatives(
        G,
        test_nodes,
        banned,
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
    
    # 创建Data对象
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_weight=edge_weight,
    )
    
    return data, test_edge_index, test_edge_label


# ==================== Training ====================


def train_gnn(
    model: nn.Module,
    data: Data,
    train_edge_index: torch.Tensor,
    train_edge_label: torch.Tensor,
    epochs: int = 200,
    lr: float = 0.01,
    device: str = 'cpu',
    val_edge_index: torch.Tensor | None = None,
    val_edge_label: torch.Tensor | None = None,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.0,
) -> nn.Module:
    """
    训练GNN模型
    """
    model = model.to(device)
    data = data.to(device)
    train_edge_index = train_edge_index.to(device)
    train_edge_label = train_edge_label.to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    
    best_pr_auc = -float("inf")
    patience = 0
    model.train()
    for epoch in tqdm(range(epochs), desc="Training", leave=False):
        optimizer.zero_grad()
        
        pred = model(data.x, data.edge_index, train_edge_index)
        loss = F.binary_cross_entropy_with_logits(pred, train_edge_label)
        
        loss.backward()
        optimizer.step()
        
        if val_edge_index is not None and val_edge_label is not None:
            model.eval()
            with torch.no_grad():
                val_pred = model(data.x, data.edge_index, val_edge_index)
                val_prob = torch.sigmoid(val_pred).cpu().numpy().tolist()
                val_label_np = val_edge_label.cpu().numpy().tolist()
            _, val_pr_auc = compute_auroc_auprc(val_label_np, val_prob)
            tqdm.write(
                f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}, "
                f"Val PR-AUC: {val_pr_auc:.4f}"
            )
            model.train()
            if val_pr_auc - best_pr_auc > early_stop_min_delta:
                best_pr_auc = val_pr_auc
                patience = 0
            else:
                patience += 1
                if early_stop_patience > 0 and patience >= early_stop_patience:
                    tqdm.write(f"Early stop at epoch {epoch+1}")
                    break
        elif (epoch + 1) % 50 == 0:
            tqdm.write(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
    
    return model


def evaluate_gnn(
    model: nn.Module,
    data: Data,
    test_edge_index: torch.Tensor,
    test_edge_label: torch.Tensor,
    test_edge_set: set[Tuple],
    node_to_idx: Dict,
    nodes: List,
    recall_grid: List[float],
    k_list: Optional[List[int]] = None,
    device: str = 'cpu',
) -> Dict:
    """
    ??GNN??
    """
    model = model.to(device)
    data = data.to(device)
    test_edge_index = test_edge_index.to(device)
    test_edge_label = test_edge_label.to(device)

    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index)
        pred = model.decode(z, test_edge_index)
        pred_prob = torch.sigmoid(pred)

    pred_prob_np = pred_prob.cpu().numpy().tolist()

    # Build pairs in original node ids
    if test_edge_index.dim() == 2 and test_edge_index.size(0) == 2:
        pairs_idx = test_edge_index.t().cpu().numpy().tolist()
    elif test_edge_index.dim() == 2 and test_edge_index.size(1) == 2:
        pairs_idx = test_edge_index.cpu().numpy().tolist()
    else:
        pairs_idx = []
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    pairs = []
    for row in pairs_idx:
        if len(row) != 2:
            continue
        u, v = row
        pairs.append((idx_to_node[u], idx_to_node[v]))

    metrics = compute_metrics(
        pairs,
        pred_prob_np,
        test_edge_set,
        recall_grid,
        k_list=k_list,
        total_nodes=nodes,
    )

    labels = [1 if tuple(sorted(p)) in test_edge_set else 0 for p in pairs]
    ranking_by_source = ranking_metrics_by_source(
        pairs,
        labels,
        pred_prob_np,
        k_list=metrics.get("k_list", []),
    )

    metrics["ranking_by_source"] = ranking_by_source
    return metrics


# ==================== Ranking Metrics ====================


def _ranking_metrics_by_source(
    edge_index: torch.Tensor,
    edge_label: torch.Tensor,
    edge_score: np.ndarray,
    k_list: List[int],
) -> Dict:
    pairs = edge_index.t().cpu().numpy().tolist()
    labels = edge_label.cpu().numpy().tolist()
    scores = edge_score.tolist()

    per_node = {}
    for (u, v), score, label in zip(pairs, scores, labels):
        entry_u = per_node.setdefault(int(u), [])
        entry_v = per_node.setdefault(int(v), [])
        entry_u.append((int(v), float(score), int(label)))
        entry_v.append((int(u), float(score), int(label)))

    metrics = {k: {"precision": [], "recall": [], "hit": [], "map": [], "ndcg": []} for k in k_list}
    mrr_list = []
    evaluated_nodes = 0

    for _, items in per_node.items():
        num_pos = sum(1 for _, _, label in items if label == 1)
        if num_pos == 0:
            continue
        evaluated_nodes += 1
        ranked = sorted(items, key=lambda x: x[1], reverse=True)
        labels_ranked = [label for _, _, label in ranked]

        first_pos = None
        for idx, label in enumerate(labels_ranked):
            if label == 1:
                first_pos = idx + 1
                break
        if first_pos is not None:
            mrr_list.append(1.0 / first_pos)
        else:
            mrr_list.append(0.0)

        for k in k_list:
            topk = labels_ranked[:k]
            hits = sum(topk)
            precision = hits / k
            recall = hits / num_pos
            hit = 1.0 if hits > 0 else 0.0

            denom = min(num_pos, k)
            if denom <= 0:
                map_k = 0.0
            else:
                running_hits = 0
                ap_sum = 0.0
                for idx, label in enumerate(topk, start=1):
                    if label == 1:
                        running_hits += 1
                        ap_sum += running_hits / idx
                map_k = ap_sum / denom

            dcg = 0.0
            for idx, label in enumerate(topk, start=1):
                if label == 1:
                    dcg += 1.0 / math.log2(idx + 1)
            idcg = sum(1.0 / math.log2(i + 1) for i in range(1, min(num_pos, k) + 1))
            ndcg = dcg / idcg if idcg > 0 else 0.0

            metrics[k]["precision"].append(precision)
            metrics[k]["recall"].append(recall)
            metrics[k]["hit"].append(hit)
            metrics[k]["map"].append(map_k)
            metrics[k]["ndcg"].append(ndcg)

    summary = {"k_list": list(k_list), "num_nodes": evaluated_nodes}
    summary["mrr"] = float(np.nanmean(mrr_list)) if mrr_list else float("nan")
    for k in k_list:
        summary[f"precision@{k}"] = float(np.nanmean(metrics[k]["precision"])) if metrics[k]["precision"] else float("nan")
        summary[f"recall@{k}"] = float(np.nanmean(metrics[k]["recall"])) if metrics[k]["recall"] else float("nan")
        summary[f"hit@{k}"] = float(np.nanmean(metrics[k]["hit"])) if metrics[k]["hit"] else float("nan")
        summary[f"map@{k}"] = float(np.nanmean(metrics[k]["map"])) if metrics[k]["map"] else float("nan")
        summary[f"ndcg@{k}"] = float(np.nanmean(metrics[k]["ndcg"])) if metrics[k]["ndcg"] else float("nan")
    return summary


# ==================== Experiment Runner ====================


def build_node2vec_features(
    G: nx.Graph,
    nodes: List,
    dim: int,
    walk_len: int,
    walks_per_node: int,
    window: int,
    p: float,
    q: float,
    seed: int,
    epochs: int,
    log_prefix: str | None = None,
) -> torch.Tensor:
    try:
        from gensim.models import Word2Vec
    except Exception as exc:
        raise RuntimeError("gensim is required for node2vec features") from exc

    rng = random.Random(seed)

    def node2vec_walk(start):
        walk = [start]
        while len(walk) < walk_len:
            cur = walk[-1]
            neighbors = list(G.neighbors(cur))
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
                elif G.has_edge(nxt, prev):
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

    walks: List[List[str]] = []
    node_list = list(nodes)
    for _ in range(walks_per_node):
        rng.shuffle(node_list)
        for node in node_list:
            walks.append(node2vec_walk(node))

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
    prev_cum_loss = 0.0
    for epoch in range(1, max(1, epochs) + 1):
        model.train(walks, total_examples=len(walks), epochs=1, compute_loss=True)
        cum_loss = model.get_latest_training_loss()
        epoch_loss = cum_loss - prev_cum_loss
        prev_cum_loss = cum_loss
        if log_prefix:
            print(f"[{log_prefix}] node2vec epoch {epoch}/{epochs} loss={epoch_loss:.6f}")

    features = []
    for node in node_list:
        features.append(model.wv[str(node)])
    return torch.tensor(features, dtype=torch.float)


def run_gnn_experiment(
    csv_path: Path,
    weight_col: str,
    model_type: str,
    test_size: float = 0.5,
    epochs: int = 200,
    hidden_dim: int = 64,
    embed_dim: int = 32,
    lr: float = 0.01,
    use_features: bool = True,
    seed: int = 42,
    use_node2vec: bool = True,
    node2vec_dim: int = 64,
    node2vec_walk_len: int = 40,
    node2vec_walks_per_node: int = 10,
    node2vec_window: int = 5,
    node2vec_p: float = 1.0,
    node2vec_q: float = 1.0,
    node2vec_epochs: int = 5,
    early_stop_patience: int = 0,
    early_stop_min_delta: float = 0.0,
    val_size: float = 0.1,
    node2vec_cache_dir: Optional[Path] = None,
    node2vec_cache_from_walk: bool = True,
) -> Dict:
    """
    运行单个GNN实验
    """
    # 设置随机种子
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # 加载数据
    df = prepare_dataset(csv_path, weight_col=weight_col)
    nodes = sorted(set(df["node1"]).union(set(df["node2"])))
    
    # 构建图
    G = nx.Graph()
    for _, row in df.iterrows():
        G.add_edge(row["node1"], row["node2"], weight=float(row["weight"]))
    G.add_nodes_from(nodes)
    
    # 划分边
    all_edges = list(df[["node1", "node2"]].itertuples(index=False, name=None))
    train_edges, test_edges = train_test_split(
        all_edges, test_size=test_size, random_state=seed
    )

    val_edges = []
    if early_stop_patience > 0 and 0.0 < val_size < 0.5:
        train_edges, val_edges = train_test_split(
            train_edges, test_size=val_size, random_state=seed
        )

    G_train = nx.Graph()
    for u, v in train_edges:
        if G.has_edge(u, v):
            w = float(G[u][v].get("weight", 1.0))
        else:
            w = 1.0
        G_train.add_edge(u, v, weight=w)
    G_train.add_nodes_from(nodes)

    test_edge_set = {tuple(sorted(edge)) for edge in test_edges}
    recall_grid = build_recall_grid(step=0.01, end=0.1)

    node_features = None
    if use_node2vec:
        cache_path = None
        if node2vec_cache_dir is not None:
            node2vec_cache_dir.mkdir(parents=True, exist_ok=True)
            if node2vec_cache_from_walk:
                cache_name = (
                    f"{csv_path.stem}_seed{seed}_test{test_size:.2f}"
                    f"_node2vec_walk.npz"
                )
            else:
                cache_name = (
                    f"{csv_path.stem}_seed{seed}_test{test_size:.2f}_val{val_size:.2f}"
                    f"_node2vec.npz"
                )
            cache_path = node2vec_cache_dir / cache_name
        if cache_path is not None and cache_path.is_file():
            cached = np.load(cache_path, allow_pickle=True)
            cached_nodes = list(cached["nodes"])
            if cached_nodes == nodes:
                node_features = torch.tensor(cached["embeddings"], dtype=torch.float)
                print(f"[Node2Vec] Loaded cached embeddings from {cache_path}")
        if node_features is None:
            raise RuntimeError(f"Node2Vec cache missing or incompatible: {cache_path}")
    
    # 准备PyG数据
    data, test_edge_index, test_edge_label = prepare_pyg_data(
        G_train,
        train_edges,
        test_edges,
        use_features=use_features,
        node_features=node_features,
        test_neg_multiplier=20,
    )
    
    # 为训练准备边标签
    node_to_idx = {node: idx for idx, node in enumerate(nodes)}
    train_pos_edges = []
    for u, v in train_edges:
        if u in node_to_idx and v in node_to_idx:
            train_pos_edges.append([node_to_idx[u], node_to_idx[v]])
    
    train_edge_index = torch.tensor(train_pos_edges, dtype=torch.long).t().contiguous()
    
    # 负采样
    num_train_pos = len(train_pos_edges)
    train_neg_edge_index = negative_sampling(
        edge_index=data.edge_index,
        num_nodes=len(nodes),
        num_neg_samples=num_train_pos,
    )
    
    # 合并正负样本
    train_edge_index = torch.cat([train_edge_index, train_neg_edge_index], dim=1)
    train_edge_label = torch.cat([
        torch.ones(num_train_pos),
        torch.zeros(num_train_pos)
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
        val_nodes = sorted(set(u for u, v in val_edges).union(set(v for u, v in val_edges)))
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
            torch.zeros(val_neg_edge_index.size(1))
        ])
    
    # 创建模型
    in_channels = data.x.size(1)
    
    if model_type == 'GCN':
        model = GCN(in_channels, hidden_dim, embed_dim)
    elif model_type == 'GraphSAGE':
        model = GraphSAGE(in_channels, hidden_dim, embed_dim)
    elif model_type == 'GAT':
        model = GAT(in_channels, hidden_dim, embed_dim, heads=4)
    elif model_type == 'GraphConv':
        model = GraphConvNet(in_channels, hidden_dim, embed_dim)
    elif model_type == 'GIN':
        model = GIN(in_channels, hidden_dim, embed_dim)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # 检测设备
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 训练
    start_time = time.time()
    model = train_gnn(
        model, data, train_edge_index, train_edge_label,
        epochs=epochs, lr=lr, device=device,
        val_edge_index=val_edge_index,
        val_edge_label=val_edge_label,
        early_stop_patience=early_stop_patience,
        early_stop_min_delta=early_stop_min_delta,
    )
    train_time = time.time() - start_time
    
    # 评估
    start_time = time.time()
    metrics = evaluate_gnn(
        model,
        data,
        test_edge_index,
        test_edge_label,
        test_edge_set=test_edge_set,
        node_to_idx=node_to_idx,
        nodes=nodes,
        recall_grid=recall_grid,
        device=device,
    )
    eval_time = time.time() - start_time
    
    metrics['auc'] = metrics.get('auroc')
    metrics['ap'] = metrics.get('auprc')
    metrics['precision@recall_mean'] = metrics.get('precision_at_recall_mean')

    metrics['train_time'] = train_time
    metrics['eval_time'] = eval_time
    
    return metrics


# ==================== Main Experiment ====================


def main():
    parser = argparse.ArgumentParser(description="GNN Baseline Experiments")
    parser.add_argument("--out-dir", type=Path, default=Path("gnn_baseline_results"))
    parser.add_argument(
        "--models",
        nargs="+",
        default=["GCN", "GraphSAGE", "GAT", "GraphConv", "GIN"],
        choices=["GCN", "GraphSAGE", "GAT", "GraphConv", "GIN"],
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--embed-dim", type=int, default=32)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    
    args.out_dir.mkdir(parents=True, exist_ok=True)
    
    # 定义数据集（与h3_hparam_sweep.py保持一致）
    datasets = {
        'within_period': [
            "data/2014data/2014AK_30.csv",
            "data/2015data/2015AK_30.csv",
        ],
    }
    
    all_results = {}
    
    for model_type in args.models:
        print(f"\n{'='*60}")
        print(f"Running {model_type}")
        print(f"{'='*60}")
        
        model_results = {}
        
        for category, files in datasets.items():
            for csv_file in files:
                csv_path = Path(csv_file)
                if not csv_path.exists():
                    print(f"Skipping {csv_path} (not found)")
                    continue
                
                print(f"\nDataset: {csv_path.name}")
                
                try:
                    metrics = run_gnn_experiment(
                        csv_path=csv_path,
                        weight_col="benecount",
                        model_type=model_type,
                        test_size=0.5,
                        epochs=args.epochs,
                        hidden_dim=args.hidden_dim,
                        embed_dim=args.embed_dim,
                        lr=args.lr,
                        seed=args.seed,
                    )
                    
                    print(f"  AUC: {metrics['auc']:.4f}")
                    print(f"  AP: {metrics['ap']:.4f}")
                    print(f"  Precision@Recall: {metrics['precision@recall_mean']:.4f}")
                    print(f"  Train time: {metrics['train_time']:.2f}s")
                    print(f"  Eval time: {metrics['eval_time']:.2f}s")
                    
                    model_results[csv_path.name] = metrics
                    
                except Exception as e:
                    print(f"  Error: {e}")
                    model_results[csv_path.name] = {'error': str(e)}
        
        all_results[model_type] = model_results
    
    # 保存结果
    output_file = args.out_dir / "gnn_baseline_results.json"
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Results saved to {output_file}")
    print(f"{'='*60}")
    
    # 生成汇总表格
    summary = []
    for model_type in args.models:
        for dataset_name, metrics in all_results[model_type].items():
            if 'error' not in metrics:
                summary.append({
                    'model': model_type,
                    'dataset': dataset_name,
                    'auc': metrics['auc'],
                    'ap': metrics['ap'],
                    'precision@recall': metrics['precision@recall_mean'],
                })
    
    summary_df = pd.DataFrame(summary)
    summary_df.to_csv(args.out_dir / "summary.csv", index=False)
    
    # 计算平均性能
    print("\nAverage Performance:")
    print(summary_df.groupby('model')[['auc', 'ap', 'precision@recall']].mean())


if __name__ == "__main__":
    main()

"""CrossSRM publication model with multi-view and support-conditioned graphs.

The architecture uses fixed graph mixing, two-hop static/learned spatial
residual propagation, last-value residual prediction, and a target adapter."""
from __future__ import annotations
from typing import Dict, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorAdapter(nn.Module):
    """Bottleneck residual adapter for node representation [B*N, D]."""

    def __init__(self, d_model: int, ratio: int = 4, alpha_init: float = 0.01):
        super().__init__()
        hidden = max(1, int(d_model) // int(ratio))
        self.down = nn.Linear(d_model, hidden)
        self.up = nn.Linear(hidden, d_model)
        self.act = nn.ReLU(inplace=True)
        self.alpha = nn.Parameter(torch.tensor(float(alpha_init), dtype=torch.float32))
        nn.init.kaiming_uniform_(self.down.weight, a=5**0.5)
        nn.init.zeros_(self.down.bias)
        nn.init.zeros_(self.up.weight)
        nn.init.zeros_(self.up.bias)

    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        r = self.up(self.act(self.down(z)))
        out = z + self.alpha * r
        aux = {
            "adapter_residual_l2": torch.mean(r**2),
            "adapter_delta_abs_mean": torch.mean(torch.abs(self.alpha * r)).detach(),
            "adapter_alpha": self.alpha.detach(),
        }
        return (out, aux)


class MultiViewTemporalGraphGenerator(nn.Module):
    """Non-parametric multi-view temporal similarity graph with learnable view weights.

    Views:
        1) corr:     correlation of raw normalized time series
        2) diff:     correlation of first-order differences
        3) lag:      max lagged correlation; row i attends to j if j leads i
        4) stats:    cosine similarity of node-level temporal statistics

    Input:
        x: [B, N, T, 1]
    Output:
        A_time: [B, N, N]

    Only a few parameters are learned: view_logits. This avoids learning a full
    adjacency matrix from scratch under few-shot target data.
    """

    def __init__(
        self,
        topk: int = 8,
        temperature: float = 0.2,
        max_lag: int = 3,
        init_weights=(0.2, 0.4, 0.3, 0.1),
    ):
        super().__init__()
        self.topk = int(topk)
        self.temperature = float(temperature)
        self.max_lag = int(max_lag)
        w = torch.tensor(list(init_weights), dtype=torch.float32)
        if w.numel() != 4:
            raise ValueError("init_weights must have 4 values: corr,diff,lag,stats")
        w = torch.clamp(w, min=1e-06)
        w = w / w.sum()
        self.view_logits = nn.Parameter(torch.log(w))

    @staticmethod
    def _time_standardize(v: torch.Tensor) -> torch.Tensor:
        return (v - v.mean(dim=-1, keepdim=True)) / v.std(
            dim=-1, keepdim=True, unbiased=False
        ).clamp_min(1e-06)

    @staticmethod
    def _node_standardize(f: torch.Tensor) -> torch.Tensor:
        return (f - f.mean(dim=1, keepdim=True)) / f.std(
            dim=1, keepdim=True, unbiased=False
        ).clamp_min(1e-06)

    @staticmethod
    def _cosine_sim(f: torch.Tensor) -> torch.Tensor:
        f = F.normalize(f.float(), dim=-1)
        return torch.einsum("bid,bjd->bij", f, f)

    def _corr(self, z: torch.Tensor) -> torch.Tensor:
        sim = torch.einsum("bit,bjt->bij", z, z) / max(1, z.size(-1))
        return sim

    def _diff_corr(self, v: torch.Tensor) -> torch.Tensor:
        if v.size(-1) <= 2:
            return torch.zeros(
                v.size(0), v.size(1), v.size(1), device=v.device, dtype=torch.float32
            )
        d = v[..., 1:] - v[..., :-1]
        dz = self._time_standardize(d)
        return self._corr(dz)

    def _lag_corr(self, z: torch.Tensor) -> torch.Tensor:
        B, N, T = z.shape
        if T <= 2 or self.max_lag <= 0:
            return torch.zeros(B, N, N, device=z.device, dtype=torch.float32)
        sims = []
        max_lag = min(self.max_lag, T - 1)
        for lag in range(1, max_lag + 1):
            zi = z[..., lag:]
            zj = z[..., :-lag]
            sim = torch.einsum("bit,bjt->bij", zi, zj) / max(1, zi.size(-1))
            sims.append(sim)
        sim = torch.stack(sims, dim=0).max(dim=0).values
        return sim

    def _stats_sim(self, v: torch.Tensor) -> torch.Tensor:
        mean = v.mean(dim=-1, keepdim=True)
        std = v.std(dim=-1, keepdim=True, unbiased=False)
        last = v[..., -1:].contiguous()
        if v.size(-1) > 1:
            d = v[..., 1:] - v[..., :-1]
            dmean = d.mean(dim=-1, keepdim=True)
            dstd = d.std(dim=-1, keepdim=True, unbiased=False)
        else:
            dmean = torch.zeros_like(mean)
            dstd = torch.zeros_like(std)
        stats = torch.cat([mean, std, last, dmean, dstd], dim=-1)
        stats = self._node_standardize(stats)
        return self._cosine_sim(stats)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if x.dim() != 4 or x.size(-1) != 1:
            raise RuntimeError(
                f"MultiViewTemporalGraphGenerator expects [B,N,T,1], got {tuple(x.shape)}"
            )
        v = x[..., 0].float()
        B, N, _ = v.shape
        z = self._time_standardize(v)
        S_corr = self._corr(z)
        S_diff = self._diff_corr(v)
        S_lag = self._lag_corr(z)
        S_stats = self._stats_sim(v)
        weights = torch.softmax(self.view_logits.float(), dim=0)
        S = weights[0] * S_corr + weights[1] * S_diff + weights[2] * S_lag + weights[3] * S_stats
        logits = S.float() / max(self.temperature, 1e-06)
        mask_value = -torch.finfo(logits.dtype).max
        if N > 1:
            eye = torch.eye(N, device=logits.device, dtype=torch.bool).view(1, N, N)
            logits = logits.masked_fill(eye, mask_value)
        if self.topk > 0 and self.topk < N:
            kk = max(1, int(self.topk))
            _, idx = torch.topk(logits, k=kk, dim=-1)
            mask = torch.zeros_like(logits, dtype=torch.bool)
            mask.scatter_(-1, idx, True)
            logits = logits.masked_fill(~mask, mask_value)
        A_float = torch.softmax(logits, dim=-1)
        A_float = A_float / (A_float.sum(dim=-1, keepdim=True) + 1e-06)
        ent = -(A_float * torch.log(A_float + 1e-12)).sum(dim=-1).mean()
        aux = {
            "ttg_A_entropy": ent.detach(),
            "ttg_A_abs_mean": torch.mean(torch.abs(A_float)).detach(),
            "ttg_A_max_mean": A_float.max(dim=-1).values.mean().detach(),
            "mv_w_corr": weights[0].detach(),
            "mv_w_diff": weights[1].detach(),
            "mv_w_lag": weights[2].detach(),
            "mv_w_stats": weights[3].detach(),
            "mv_S_corr_mean": S_corr.mean().detach(),
            "mv_S_diff_mean": S_diff.mean().detach(),
            "mv_S_lag_mean": S_lag.mean().detach(),
            "mv_S_stats_mean": S_stats.mean().detach(),
        }
        return (A_float.to(dtype=x.dtype), aux)


class TaskConditionedSpatialGraphLearner(nn.Module):
    """Few-shot task-conditioned latent spatial graph learner.

    The final graph is generated at node-pair level from support-set node
    embeddings and relation features. Hand-crafted temporal similarities are
    used only as candidate-edge constraints / relation features, not as the
    final adjacency matrix.

    support_x: [S, N, T, 1]
    A_static:  [N, N]
    returns:   A_task [N, N]
    """

    def __init__(
        self,
        topm: int = 24,
        topk: int = 8,
        temperature: float = 0.2,
        graph_dim: int = 64,
        rel_hidden: int = 32,
        max_lag: int = 3,
        beta_rel: float = 1.0,
        qk_scale: float = 1.0,
        static_candidate_weight: float = 0.5,
    ):
        super().__init__()
        self.topm = int(topm)
        self.topk = int(topk)
        self.temperature = float(temperature)
        self.graph_dim = int(graph_dim)
        self.max_lag = int(max_lag)
        self.beta_rel = float(beta_rel)
        self.qk_scale = float(qk_scale)
        self.static_candidate_weight = float(static_candidate_weight)
        self.node_encoder = nn.Sequential(
            nn.LayerNorm(6),
            nn.Linear(6, self.graph_dim),
            nn.GELU(),
            nn.Linear(self.graph_dim, self.graph_dim),
        )
        self.q_proj = nn.Linear(self.graph_dim, self.graph_dim)
        self.k_proj = nn.Linear(self.graph_dim, self.graph_dim)
        self.rel_mlp = nn.Sequential(
            nn.LayerNorm(5), nn.Linear(5, int(rel_hidden)), nn.GELU(), nn.Linear(int(rel_hidden), 1)
        )
        nn.init.zeros_(self.rel_mlp[-1].weight)
        nn.init.zeros_(self.rel_mlp[-1].bias)

    @staticmethod
    def _time_standardize(v: torch.Tensor) -> torch.Tensor:
        return (v - v.mean(dim=-1, keepdim=True)) / v.std(
            dim=-1, keepdim=True, unbiased=False
        ).clamp_min(1e-06)

    @staticmethod
    def _node_standardize(f: torch.Tensor) -> torch.Tensor:
        return (f - f.mean(dim=0, keepdim=True)) / f.std(
            dim=0, keepdim=True, unbiased=False
        ).clamp_min(1e-06)

    @staticmethod
    def _row_normalize(A: torch.Tensor) -> torch.Tensor:
        A = torch.clamp(A.float(), min=0.0)
        return A / (A.sum(dim=-1, keepdim=True) + 1e-06)

    def _support_node_stats(self, v: torch.Tensor) -> torch.Tensor:
        S, N, T = v.shape
        flat = v.permute(1, 0, 2).reshape(N, S * T)
        mean = flat.mean(dim=-1, keepdim=True)
        std = flat.std(dim=-1, keepdim=True, unbiased=False)
        last_mean = v[..., -1].mean(dim=0, keepdim=True).transpose(0, 1)
        if T > 1:
            d = v[..., 1:] - v[..., :-1]
            dflat = d.permute(1, 0, 2).reshape(N, S * (T - 1))
            dmean = dflat.mean(dim=-1, keepdim=True)
            dstd = dflat.std(dim=-1, keepdim=True, unbiased=False)
            recent = (
                (v[..., -1] - v[..., max(0, T - 1 - min(12, T - 1))])
                .mean(dim=0, keepdim=True)
                .transpose(0, 1)
            )
        else:
            dmean = torch.zeros_like(mean)
            dstd = torch.zeros_like(std)
            recent = torch.zeros_like(mean)
        stats = torch.cat([mean, std, last_mean, dmean, dstd, recent], dim=-1)
        return self._node_standardize(stats.float())

    def _corr_from_series(self, a: torch.Tensor, b: torch.Tensor | None = None) -> torch.Tensor:
        if b is None:
            b = a
        return torch.einsum("it,jt->ij", a, b) / max(1, a.size(-1))

    def _relation_features(
        self, v: torch.Tensor, A_static: torch.Tensor | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        S, N, T = v.shape
        z = self._time_standardize(v.float())
        S_corr_list, S_diff_list, S_lag_list, S_stats_list = ([], [], [], [])
        for s in range(S):
            zs = z[s]
            S_corr = self._corr_from_series(zs)
            if T > 2:
                d = v[s, :, 1:] - v[s, :, :-1]
                dz = self._time_standardize(d)
                S_diff = self._corr_from_series(dz)
            else:
                S_diff = torch.zeros(N, N, device=v.device, dtype=torch.float32)
            if T > 2 and self.max_lag > 0:
                lag_sims = []
                for lag in range(1, min(self.max_lag, T - 1) + 1):
                    lag_sims.append(self._corr_from_series(zs[:, lag:], zs[:, :-lag]))
                S_lag = torch.stack(lag_sims, dim=0).max(dim=0).values
            else:
                S_lag = torch.zeros(N, N, device=v.device, dtype=torch.float32)
            mean = v[s].mean(dim=-1, keepdim=True)
            std = v[s].std(dim=-1, keepdim=True, unbiased=False)
            last = v[s, :, -1:].contiguous()
            if T > 1:
                d = v[s, :, 1:] - v[s, :, :-1]
                dmean = d.mean(dim=-1, keepdim=True)
                dstd = d.std(dim=-1, keepdim=True, unbiased=False)
            else:
                dmean = torch.zeros_like(mean)
                dstd = torch.zeros_like(std)
            stats = torch.cat([mean, std, last, dmean, dstd], dim=-1)
            stats = self._node_standardize(stats.float())
            stats = F.normalize(stats, dim=-1)
            S_stats = torch.einsum("id,jd->ij", stats, stats)
            S_corr_list.append(S_corr)
            S_diff_list.append(S_diff)
            S_lag_list.append(S_lag)
            S_stats_list.append(S_stats)
        S_corr = torch.stack(S_corr_list, dim=0).mean(dim=0)
        S_diff = torch.stack(S_diff_list, dim=0).mean(dim=0)
        S_lag = torch.stack(S_lag_list, dim=0).mean(dim=0)
        S_stats = torch.stack(S_stats_list, dim=0).mean(dim=0)
        if A_static is None:
            A_s = torch.zeros_like(S_corr)
        else:
            A_s = self._row_normalize(A_static).to(device=v.device, dtype=torch.float32)
        edge_attr = torch.stack([S_corr, S_diff, S_lag, S_stats, A_s], dim=-1)
        candidate_score = (
            S_corr + S_lag + 0.5 * S_diff + 0.25 * S_stats + self.static_candidate_weight * A_s
        )
        return (edge_attr, candidate_score)

    def forward(
        self, support_x: torch.Tensor, A_static: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        if support_x.dim() != 4 or support_x.size(-1) != 1:
            raise RuntimeError(
                f"TaskGraphLearner expects support_x [S,N,T,1], got {tuple(support_x.shape)}"
            )
        v = support_x[..., 0].float()
        S, N, T = v.shape
        node_stats = self._support_node_stats(v)
        z = self.node_encoder(node_stats)
        q = F.normalize(self.q_proj(z), dim=-1)
        k = F.normalize(self.k_proj(z), dim=-1)
        qk = torch.einsum("id,jd->ij", q, k) / math.sqrt(max(1, self.graph_dim))
        edge_attr, cand_score = self._relation_features(v, A_static)
        rel_bias = self.rel_mlp(edge_attr).squeeze(-1)
        logits = self.qk_scale * qk + self.beta_rel * rel_bias
        mask_value = -torch.finfo(logits.dtype).max
        if N > 1:
            eye = torch.eye(N, device=logits.device, dtype=torch.bool)
            logits = logits.masked_fill(eye, mask_value)
            cand_score = cand_score.masked_fill(eye, mask_value)
        topm = self.topm if self.topm > 0 else N
        topm = min(max(1, int(topm)), N - 1 if N > 1 else 1)
        if topm < N:
            _, idx = torch.topk(cand_score.float(), k=topm, dim=-1)
            cand_mask = torch.zeros_like(logits, dtype=torch.bool)
            cand_mask.scatter_(-1, idx, True)
            logits = logits.masked_fill(~cand_mask, mask_value)
        topk = min(max(1, int(self.topk)), N - 1 if N > 1 else 1)
        if topk < N:
            _, idx = torch.topk(logits.float(), k=topk, dim=-1)
            final_mask = torch.zeros_like(logits, dtype=torch.bool)
            final_mask.scatter_(-1, idx, True)
            logits = logits.masked_fill(~final_mask, mask_value)
        A = torch.softmax(logits.float() / max(self.temperature, 1e-06), dim=-1)
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-06)
        ent = -(A * torch.log(A + 1e-12)).sum(dim=-1).mean()
        aux = {
            "task_A_entropy": ent.detach(),
            "task_A_max_mean": A.max(dim=-1).values.mean().detach(),
            "task_qk_abs_mean": qk.abs().mean().detach(),
            "task_rel_bias_abs_mean": rel_bias.abs().mean().detach(),
            "task_candidate_topm": torch.tensor(float(topm), device=A.device),
            "task_final_topk": torch.tensor(float(topk), device=A.device),
        }
        return (A, aux)


class TaskGraphReliabilityGate(nn.Module):
    """Mix stable and support-conditioned graphs using a fixed ratio."""

    def __init__(self, mix_init: float = 0.3):
        super().__init__()
        self.mix_init = float(mix_init)

    def forward(
        self, x: torch.Tensor, A_stable: torch.Tensor, A_task: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B = x.size(0)
        eta = torch.full((B, 1, 1), self.mix_init, device=x.device, dtype=torch.float32)
        A = (1.0 - eta) * A_stable.float() + eta * A_task.float()
        A = A / (A.sum(dim=-1, keepdim=True) + 1e-06)
        aux = {
            "task_mix_eta_mean": eta.mean().detach(),
            "task_mix_eta_min": eta.min().detach(),
            "task_mix_eta_max": eta.max().detach(),
            "task_mix_gap_abs_mean": (A_task.float() - A_stable.float()).abs().mean().detach(),
            "task_mix_final_entropy": (-(A * torch.log(A + 1e-12)).sum(dim=-1).mean()).detach(),
            "task_mix_final_max_mean": A.max(dim=-1).values.mean().detach(),
        }
        return (A.to(dtype=A_stable.dtype), aux)


class NonlinearGraphResidualBranch(nn.Module):
    """Lag-aware nonlinear graph residual branch.

    It does NOT smooth the final scalar prediction directly. Instead it first
    propagates high-dimensional node representations through both the learned
    functional graph and the static road graph, then uses an MLP to predict a
    horizon-wise residual correction.

    Input:
        h:        [B, N, D]
        A_time:   [B, N, N]
        A_static: [N, N] or [B, N, N], optional
    Output:
        r_graph:  [B, N, H]
    """

    def __init__(self, d_model: int, horizon: int, hidden_dim: int = 128, dropout: float = 0.1):
        super().__init__()
        self.d_model = int(d_model)
        self.horizon = int(horizon)
        self.hidden_dim = int(hidden_dim)
        mult = 5
        self.net = nn.Sequential(
            nn.LayerNorm(mult * self.d_model),
            nn.Linear(mult * self.d_model, self.hidden_dim),
            nn.GELU(),
            nn.Dropout(float(dropout)),
            nn.Linear(self.hidden_dim, self.horizon),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    @staticmethod
    def normalize_A(A: torch.Tensor) -> torch.Tensor:
        if A.dim() not in {2, 3}:
            raise RuntimeError(
                f"GraphResidualBranch expects A [N,N] or [B,N,N], got {tuple(A.shape)}"
            )
        A = torch.clamp(A.float(), min=0.0)
        return A / (A.sum(dim=-1, keepdim=True) + 1e-06)

    @staticmethod
    def propagate(A: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        A = NonlinearGraphResidualBranch.normalize_A(A).to(device=h.device, dtype=h.dtype)
        if A.dim() == 2:
            return torch.einsum("ij,bjd->bid", A, h)
        return torch.einsum("bij,bjd->bid", A, h)

    def forward(
        self, h: torch.Tensor, A_time: torch.Tensor, A_static: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        time1 = self.propagate(A_time, h)
        time2 = self.propagate(A_time, time1)
        if A_static is not None:
            static1 = self.propagate(A_static, h)
            static2 = self.propagate(A_static, static1)
        else:
            static1 = torch.zeros_like(h)
            static2 = torch.zeros_like(h)
        feat = torch.cat([h, time1 - h, time2 - h, static1 - h, static2 - h], dim=-1)
        r_graph = self.net(feat)
        aux = {
            "graph_branch_abs_mean": torch.mean(torch.abs(r_graph)).detach(),
            "graph_time_gap_abs_mean": torch.mean(torch.abs(time1 - h)).detach(),
            "graph_static_gap_abs_mean": torch.mean(torch.abs(static1 - h)).detach(),
        }
        return (r_graph, aux)


class CrossSRM(nn.Module):
    """Publication model: last value + temporal residual + gated spatial residual.

    Input x: [B, N, T, 1]. Output: normalized predictions [B, N, H].
    A_task must be built from support windows, then reused for query prediction."""

    def __init__(
        self,
        input_len: int = 288,
        in_dim: int = 1,
        out_dim: int = 12,
        patch_len: int = 12,
        stride: int = 12,
        d_model: int = 128,
        n_layers: int = 3,
        n_heads: int = 4,
        ffn_dim: int = 256,
        dropout: float = 0.1,
        adapter_ratio: int = 4,
        alpha_init: float = 0.01,
        graph_topk: int = 8,
        graph_temperature: float = 0.2,
        graph_branch_hidden: int = 128,
        graph_branch_dropout: float = 0.1,
        graph_gate_init: float = 0.01,
        graph_gate_max: float = 0.3,
        task_graph_topm: int = 24,
        task_graph_dim: int = 64,
        task_graph_rel_hidden: int = 32,
        task_graph_beta: float = 1.0,
        task_graph_qk_scale: float = 1.0,
        task_graph_static_candidate_weight: float = 0.5,
        task_graph_temperature: float = 0.05,
        task_graph_mix: float = 0.3,
        mv_max_lag: int = 3,
        mv_init_weights=(0.2, 0.4, 0.3, 0.1),
    ):
        super().__init__()
        if input_len < patch_len:
            raise ValueError(f"input_len={input_len} must be >= patch_len={patch_len}")
        self.input_len = int(input_len)
        self.in_dim = int(in_dim)
        self.out_dim = int(out_dim)
        self.patch_len = int(patch_len)
        self.stride = int(stride)
        self.d_model = int(d_model)
        self.graph_gate_max = float(graph_gate_max)
        self.num_patches = 1 + (self.input_len - self.patch_len) // self.stride
        if self.num_patches <= 0:
            raise ValueError("num_patches <= 0; check input_len/patch_len/stride")
        self.patch_proj = nn.Linear(self.patch_len * self.in_dim, self.d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, self.num_patches, self.d_model))
        self.dropout = nn.Dropout(float(dropout))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.d_model,
            nhead=int(n_heads),
            dim_feedforward=int(ffn_dim),
            dropout=float(dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(n_layers))
        self.norm = nn.LayerNorm(self.d_model)
        self.graph_generator = MultiViewTemporalGraphGenerator(
            topk=graph_topk,
            temperature=graph_temperature,
            max_lag=mv_max_lag,
            init_weights=mv_init_weights,
        )
        self.task_graph_learner = TaskConditionedSpatialGraphLearner(
            topm=task_graph_topm,
            topk=graph_topk,
            temperature=graph_temperature
            if task_graph_temperature is None or float(task_graph_temperature) <= 0
            else float(task_graph_temperature),
            graph_dim=task_graph_dim,
            rel_hidden=task_graph_rel_hidden,
            max_lag=mv_max_lag,
            beta_rel=task_graph_beta,
            qk_scale=task_graph_qk_scale,
            static_candidate_weight=task_graph_static_candidate_weight,
        )
        self.task_graph_mix_gate = TaskGraphReliabilityGate(mix_init=task_graph_mix)
        self.graph_residual_branch = NonlinearGraphResidualBranch(
            d_model=self.d_model,
            horizon=self.out_dim,
            hidden_dim=graph_branch_hidden,
            dropout=graph_branch_dropout,
        )
        ratio_gate = max(
            -0.999, min(0.999, float(graph_gate_init) / max(self.graph_gate_max, 1e-06))
        )
        self.raw_graph_gate = nn.Parameter(
            torch.full((self.out_dim,), math.atanh(ratio_gate), dtype=torch.float32)
        )
        self.head = nn.Linear(self.d_model, self.out_dim)
        self.target_adapter = VectorAdapter(self.d_model, adapter_ratio, alpha_init)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        nn.init.xavier_uniform_(self.patch_proj.weight)
        nn.init.zeros_(self.patch_proj.bias)
        nn.init.xavier_uniform_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def set_target_adapter(self, adapter_ratio: int = 4, alpha_init: float = 0.01) -> None:
        self.target_adapter = VectorAdapter(self.d_model, adapter_ratio, alpha_init).to(
            next(self.parameters()).device
        )

    def load_state_dict(self, state_dict, strict: bool = True):
        """Load release weights or a matching full-model checkpoint."""
        if "graph_gamma" in state_dict:
            cleaned = state_dict.copy()
            cleaned.pop("graph_gamma")
            if hasattr(state_dict, "_metadata"):
                cleaned._metadata = state_dict._metadata
            state_dict = cleaned
        return super().load_state_dict(state_dict, strict=strict)

    def _patchify(self, x_bn: torch.Tensor) -> torch.Tensor:
        patches = x_bn.unfold(dimension=1, size=self.patch_len, step=self.stride)
        patches = patches.permute(0, 1, 3, 2).contiguous()
        return patches.view(patches.size(0), patches.size(1), self.patch_len * self.in_dim)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B, N, T, F_in = x.shape
        if T != self.input_len:
            if T < self.patch_len:
                raise RuntimeError(f"Input length {T} is shorter than patch_len {self.patch_len}")
            if T > self.input_len:
                x = x[:, :, -self.input_len :, :].contiguous()
                B, N, T, F_in = x.shape
        if F_in != self.in_dim:
            raise RuntimeError(f"Expected in_dim={self.in_dim}, got {F_in}")
        x_bn = x.reshape(B * N, T, F_in)
        p = self._patchify(x_bn)
        z = self.patch_proj(p)
        z = self.dropout(z + self.pos_emb[:, : z.size(1), :])
        z = self.encoder(z)
        h = z[:, -1, :]
        h = self.norm(h)
        return h.view(B, N, self.d_model)

    def build_task_graph(
        self, support_x: torch.Tensor, A_static: torch.Tensor | None = None
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        return self.task_graph_learner(support_x, A_static)

    def mix_task_graph(
        self, x: torch.Tensor, A_task: torch.Tensor, dtype: torch.dtype
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Combine the query-window graph with a graph built from support only."""
        B = x.size(0)
        if A_task is None:
            raise ValueError(
                "A_task is required; construct it with build_task_graph(support_x, A_static)."
            )
        if A_task.dim() == 2:
            A_task_b = A_task.to(device=x.device, dtype=dtype).unsqueeze(0).expand(B, -1, -1)
        elif A_task.dim() == 3:
            if A_task.size(0) == 1:
                A_task_b = A_task.to(device=x.device, dtype=dtype).expand(B, -1, -1)
            elif A_task.size(0) == B:
                A_task_b = A_task.to(device=x.device, dtype=dtype)
            else:
                raise RuntimeError(f"A_task batch mismatch: A_task={tuple(A_task.shape)}, B={B}")
        else:
            raise RuntimeError(f"A_task must be [N,N] or [B,N,N], got {tuple(A_task.shape)}")
        A_stable, stable_aux = self.graph_generator(x)
        A_time, mix_aux = self.task_graph_mix_gate(x, A_stable.to(dtype=dtype), A_task_b)
        aux = {
            "ttg_A_entropy": (
                -(A_time.float() * torch.log(A_time.float() + 1e-12)).sum(dim=-1).mean()
            ).detach(),
            "ttg_A_abs_mean": torch.mean(torch.abs(A_time.float())).detach(),
            "ttg_A_max_mean": A_time.float().max(dim=-1).values.mean().detach(),
            "task_graph_used": torch.tensor(1.0, device=x.device),
            "task_A_entropy_in_model": (
                -(A_task_b.float() * torch.log(A_task_b.float() + 1e-12)).sum(dim=-1).mean()
            ).detach(),
            "task_A_max_in_model": A_task_b.float().max(dim=-1).values.mean().detach(),
            "stable_A_entropy": stable_aux.get(
                "ttg_A_entropy", torch.tensor(0.0, device=x.device)
            ).detach(),
            "stable_A_max_mean": stable_aux.get(
                "ttg_A_max_mean", torch.tensor(0.0, device=x.device)
            ).detach(),
            "A_time": A_time,
        }
        aux.update(mix_aux)
        return (A_time, aux)

    def forward(
        self,
        x: torch.Tensor,
        use_target_adapter: bool = False,
        A_static: torch.Tensor | None = None,
        A_task: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, N, _, _ = x.shape
        last_value = x[:, :, -1, 0].contiguous()
        h = self.encode(x)
        A_time, aux = self.mix_task_graph(x, A_task, h.dtype)
        h_flat = h.reshape(B * N, self.d_model)
        if use_target_adapter:
            h_flat, ad_aux = self.target_adapter(h_flat)
            aux.update(ad_aux)
        h_adapted = h_flat.view(B, N, self.d_model)
        r_temporal = self.head(h_flat).view(B, N, self.out_dim)
        A_for_branch = A_time
        r_graph, br_aux = self.graph_residual_branch(h_adapted, A_for_branch, A_static)
        gate = self.graph_gate_max * torch.tanh(self.raw_graph_gate).view(1, 1, -1)
        y = r_temporal + gate * r_graph
        aux.update(br_aux)
        aux["graph_gate_abs_mean"] = torch.mean(torch.abs(gate)).detach()
        aux["graph_gate_max_abs"] = torch.max(torch.abs(gate)).detach()
        aux["graph_residual_delta_abs_mean"] = torch.mean(torch.abs(gate * r_graph)).detach()
        for i in range(self.out_dim):
            aux[f"graph_gate_h{i + 1:02d}"] = gate.view(-1)[i].detach()
        y = y + last_value.unsqueeze(-1)
        aux["lv_residual_enabled"] = torch.tensor(1.0, device=y.device)
        aux["last_value_abs_mean"] = torch.mean(torch.abs(last_value)).detach()
        return (y, aux)


__all__ = [
    "CrossSRM",
    "VectorAdapter",
    "MultiViewTemporalGraphGenerator",
    "TaskConditionedSpatialGraphLearner",
    "TaskGraphReliabilityGate",
    "NonlinearGraphResidualBranch",
]

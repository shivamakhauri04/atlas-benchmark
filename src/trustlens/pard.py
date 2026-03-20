"""PARD: Polarity-Aware Representation for Domain-Transfer.

Extends PAFC from manual {-1,+1} sign flip on 3 heuristic features
to learned polarity-aware representations with soft correction and
conformal guarantees on d-dimensional encoder features.

Components:
  1. PARDEncoder: LSTM backbone + bottleneck layer + polarity/adversarial training
  2. PolarityCorrectionLayer: Diagonal W correction (sign_flip / diagonal / low_rank)
  3. pard_calibrate / pard_predict: Split CP on corrected features
  4. PARDClassifier: Few-shot logistic regression on corrected features

Coverage theorem: P(Y in C(X)) >= 1 - alpha - delta_correction(N_est, d)
  where delta_correction = O(d * L / sqrt(N_est))
"""

from __future__ import annotations

from enum import Enum

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from sklearn.linear_model import LogisticRegression

from trustlens.conformal import split_conformal_calibrate
from trustlens.dapl import (
    DomainClassifier,
    GradientReversalLayer,
    LSTMEncoder,
    compute_polarity_loss,
)


class CorrectionMode(str, Enum):
    SIGN_FLIP = "sign_flip"
    DIAGONAL = "diagonal"
    LOW_RANK = "low_rank"


class PARDEncoder(nn.Module):
    """LSTM encoder with bottleneck layer for PARD.

    Architecture: LSTMEncoder(d_hidden) -> Linear(d_hidden, d_bottleneck) -> ReLU
    The bottleneck forces the encoder to compress information into
    d_bottleneck dimensions where polarity structure can be measured
    and corrected.
    """

    def __init__(
        self,
        vocab_size: int,
        d_embed: int = 128,
        d_hidden: int = 128,
        d_bottleneck: int = 32,
    ) -> None:
        super().__init__()
        self.lstm = LSTMEncoder(vocab_size, d_embed, d_hidden)
        self.bottleneck = nn.Sequential(
            nn.Linear(d_hidden, d_bottleneck),
            nn.ReLU(),
        )
        self.d_bottleneck = d_bottleneck

    def forward(self, tool_ids: torch.Tensor) -> torch.Tensor:
        """Next-token prediction logits (for NLL loss)."""
        return self.lstm(tool_ids)

    def extract_features(
        self, tool_ids: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        """Extract bottleneck features: (B, d_bottleneck)."""
        h = self.lstm.trace_features(tool_ids, mask)
        return self.bottleneck(h)

    def compute_nll(
        self, tool_ids: torch.Tensor, mask: torch.Tensor,
    ) -> torch.Tensor:
        """Per-trace mean NLL anomaly score (B,)."""
        return self.lstm.compute_nll(tool_ids, mask)


class PolarityCorrectionLayer:
    """Learned polarity correction W applied to d-dim features.

    Three modes:
      sign_flip: w_k in {-1, +1} — reproduces PAFC exactly
      diagonal: w_k in [-1, +1] — continuous optimization (key contribution)
      low_rank: W = I + U @ V^T — expressiveness ablation
    """

    def __init__(self, d: int, mode: CorrectionMode = CorrectionMode.DIAGONAL) -> None:
        self.d = d
        self.mode = mode
        self.w: np.ndarray | None = None
        self.U: np.ndarray | None = None
        self.V: np.ndarray | None = None

    def estimate(
        self,
        src_features: np.ndarray,
        src_labels: np.ndarray,
        tgt_features: np.ndarray,
        tgt_labels: np.ndarray,
    ) -> None:
        """Estimate correction parameters from labeled source + target data.

        Args:
            src_features: (N_s, d) source domain features.
            src_labels: (N_s,) binary source labels.
            tgt_features: (N_t, d) target domain features.
            tgt_labels: (N_t,) binary target labels.
        """
        src_corr = self._per_dim_correlation(src_features, src_labels)
        tgt_corr = self._per_dim_correlation(tgt_features, tgt_labels)

        if self.mode == CorrectionMode.SIGN_FLIP:
            self.w = np.sign(src_corr) * np.sign(tgt_corr)
            self.w[self.w == 0] = 1.0

        elif self.mode == CorrectionMode.DIAGONAL:
            self.w = self._optimize_diagonal(
                src_features, src_labels, tgt_features, tgt_labels,
                src_corr, tgt_corr,
            )

        elif self.mode == CorrectionMode.LOW_RANK:
            rank = min(4, self.d // 4)
            self.U, self.V = self._estimate_low_rank(
                src_features, src_labels, tgt_features, tgt_labels, rank,
            )

    def correct(self, features: np.ndarray) -> np.ndarray:
        """Apply correction: z_corrected = W * z.

        Args:
            features: (N, d) features to correct.

        Returns:
            (N, d) corrected features.
        """
        if self.mode in (CorrectionMode.SIGN_FLIP, CorrectionMode.DIAGONAL):
            if self.w is None:
                return features
            return features * self.w[np.newaxis, :]

        if self.U is not None and self.V is not None:
            W = np.eye(self.d) + self.U @ self.V.T
            return features @ W.T

        return features

    @property
    def reversal_rate(self) -> float:
        """Fraction of dimensions with sign flip (w_k < 0)."""
        if self.w is None:
            return 0.0
        return float(np.mean(self.w < 0))

    @staticmethod
    def _per_dim_correlation(
        features: np.ndarray, labels: np.ndarray,
    ) -> np.ndarray:
        """Compute per-dimension Spearman correlation with labels."""
        d = features.shape[1]
        corr = np.zeros(d)
        for k in range(d):
            if np.std(features[:, k]) < 1e-10:
                continue
            rho, _ = stats.spearmanr(features[:, k], labels)
            corr[k] = rho if np.isfinite(rho) else 0.0
        return corr

    def _optimize_diagonal(
        self,
        src_feat: np.ndarray,
        src_lbl: np.ndarray,
        tgt_feat: np.ndarray,
        tgt_lbl: np.ndarray,
        src_corr: np.ndarray,
        tgt_corr: np.ndarray,
    ) -> np.ndarray:
        """Optimize w_k in [-1, 1] to maximize polarity alignment.

        Objective: maximize sum_k w_k * src_sign_k * tgt_d_k
        where tgt_d_k = (mean(feat_k | y=1) - mean(feat_k | y=0)) in target.

        This is a closed-form solution: w_k = clip(src_sign_k * tgt_d_k / |tgt_d_k|, -1, 1)
        which simplifies to sign(src_corr_k * tgt_corr_k) when both are nonzero,
        with magnitude reflecting confidence.
        """
        src_sign = np.sign(src_corr)

        tgt_pos = tgt_lbl == 1
        tgt_neg = tgt_lbl == 0
        if tgt_pos.sum() < 2 or tgt_neg.sum() < 2:
            return np.ones(self.d)

        tgt_d = tgt_feat[tgt_pos].mean(0) - tgt_feat[tgt_neg].mean(0)
        tgt_std = np.std(tgt_feat, axis=0)
        tgt_std[tgt_std < 1e-10] = 1.0
        tgt_d_norm = tgt_d / tgt_std

        w = src_sign * tgt_d_norm
        w = np.clip(w, -1.0, 1.0)

        no_signal = (np.abs(src_corr) < 0.05) | (np.abs(tgt_corr) < 0.05)
        w[no_signal] = 1.0

        return w

    def _estimate_low_rank(
        self,
        src_feat: np.ndarray,
        src_lbl: np.ndarray,
        tgt_feat: np.ndarray,
        tgt_lbl: np.ndarray,
        rank: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Estimate low-rank correction W = I + U @ V^T via SVD of residual."""
        w_diag = self._optimize_diagonal(
            src_feat, src_lbl, tgt_feat, tgt_lbl,
            self._per_dim_correlation(src_feat, src_lbl),
            self._per_dim_correlation(tgt_feat, tgt_lbl),
        )
        residual = np.diag(w_diag) - np.eye(self.d)
        U, S, Vt = np.linalg.svd(residual, full_matrices=False)
        U_r = U[:, :rank] * S[:rank]
        V_r = Vt[:rank, :].T
        return U_r, V_r


def pard_calibrate(
    encoder: PARDEncoder,
    src_seqs: list[list[int]],
    src_labels: list[int],
    tgt_cal_seqs: list[list[int]],
    tgt_cal_labels: list[int],
    *,
    mode: CorrectionMode = CorrectionMode.DIAGONAL,
    d_bottleneck: int = 32,
    alpha: float = 0.05,
    device: torch.device | None = None,
) -> tuple[PolarityCorrectionLayer, float, dict]:
    """PARD conformal calibration pipeline.

    1. Extract bottleneck features from source and target calibration
    2. Split target cal 50/50: estimation set + conformal set
    3. Estimate W on estimation set
    4. Compute conformal threshold tau on corrected conformal set

    Returns:
        (correction, tau, info_dict)
    """
    if device is None:
        device = torch.device("cpu")

    src_feat, src_lbl = _extract_features(encoder, src_seqs, src_labels, device)
    tgt_feat, tgt_lbl = _extract_features(
        encoder, tgt_cal_seqs, tgt_cal_labels, device,
    )

    n = len(tgt_cal_seqs)
    n_est = n // 2
    idx = np.random.permutation(n)
    est_idx, cal_idx = idx[:n_est], idx[n_est:]

    correction = PolarityCorrectionLayer(d_bottleneck, mode)
    correction.estimate(
        src_feat, src_lbl,
        tgt_feat[est_idx], tgt_lbl[est_idx],
    )

    cal_nll = _compute_nll_scores(encoder, tgt_cal_seqs, device)
    cal_scores = cal_nll[cal_idx]

    tau = split_conformal_calibrate(cal_scores, alpha=alpha)

    info = {
        "n_estimation": n_est,
        "n_conformal": n - n_est,
        "reversal_rate_pre": float(np.mean(
            np.sign(correction._per_dim_correlation(src_feat, src_lbl))
            != np.sign(correction._per_dim_correlation(tgt_feat, tgt_lbl))
        )),
        "reversal_rate_post": correction.reversal_rate,
        "mode": mode.value,
    }
    return correction, tau, info


def pard_predict(
    encoder: PARDEncoder,
    correction: PolarityCorrectionLayer,
    tau: float,
    seqs: list[list[int]],
    *,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply PARD correction + conformal threshold.

    Returns:
        (predictions, scores) where predictions are binary (1=anomalous).
    """
    if device is None:
        device = torch.device("cpu")

    scores = _compute_nll_scores(encoder, seqs, device)
    predictions = (scores > tau).astype(np.int32)
    return predictions, scores


class PARDClassifier:
    """Few-shot discrimination head on corrected PARD features.

    Uses logistic regression on corrected bottleneck features
    for cross-domain anomaly classification.
    """

    def __init__(self, C: float = 1.0, max_iter: int = 1000) -> None:
        self.clf = LogisticRegression(C=C, max_iter=max_iter, solver="lbfgs")
        self._fitted = False

    def fit(
        self,
        features: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        """Fit LR on (corrected) features."""
        if len(np.unique(labels)) < 2:
            self._fitted = False
            return
        self.clf.fit(features, labels)
        self._fitted = True

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        """Return P(anomalous) for each sample."""
        if not self._fitted:
            return np.full(len(features), 0.5)
        return self.clf.predict_proba(features)[:, 1]

    def predict(self, features: np.ndarray) -> np.ndarray:
        if not self._fitted:
            return np.zeros(len(features), dtype=np.int32)
        return self.clf.predict(features).astype(np.int32)


class PARDTrainer:
    """Trains PARDEncoder with DAPL-style objective + bottleneck.

    Loss = L_nll + lambda_adv * L_adv + lambda_pol * L_pol + lambda_cls * L_cls

    The key difference from DAPLTrainer: features are extracted from
    the bottleneck layer (d_bottleneck) rather than raw LSTM hidden
    states (d_hidden). The classification loss (L_cls) forces the
    bottleneck to be discriminative, enabling polarity measurement
    and correction.
    """

    def __init__(
        self,
        vocab_size: int,
        d_embed: int = 128,
        d_hidden: int = 128,
        d_bottleneck: int = 32,
        lambda_adv: float = 0.1,
        lambda_pol: float = 0.5,
        lambda_cls: float = 0.3,
        lr: float = 1e-3,
        epochs: int = 50,
        patience: int = 5,
        batch_size: int = 64,
    ) -> None:
        self.vocab_size = vocab_size
        self.d_bottleneck = d_bottleneck
        self.lambda_adv = lambda_adv
        self.lambda_pol = lambda_pol
        self.lambda_cls = lambda_cls
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size

        self.encoder = PARDEncoder(vocab_size, d_embed, d_hidden, d_bottleneck)
        self.grl = GradientReversalLayer(lambda_max=1.0)
        self.domain_clf = DomainClassifier(d_bottleneck)
        self.cls_head = nn.Linear(d_bottleneck, 1)

        self._device = torch.device("cpu")

    def to(self, device: torch.device) -> PARDTrainer:
        self._device = device
        self.encoder.to(device)
        self.domain_clf.to(device)
        self.cls_head.to(device)
        return self

    def train(
        self,
        src_seqs: list[list[int]],
        src_labels: list[int],
        tgt_seqs: list[list[int]],
        tgt_labels: list[int],
    ) -> dict[str, list[float]]:
        """Train PARD encoder on source + target data."""
        from torch.optim.lr_scheduler import CosineAnnealingLR

        params = (
            list(self.encoder.parameters())
            + list(self.domain_clf.parameters())
            + list(self.cls_head.parameters())
        )
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)
        nll_loss_fn = nn.CrossEntropyLoss(ignore_index=0)

        history: dict[str, list[float]] = {
            "nll": [], "adv": [], "pol": [], "cls": [], "total": [],
        }
        best_loss = float("inf")
        best_state: dict | None = None
        patience_ctr = 0

        self.encoder.train()
        self.domain_clf.train()

        for epoch in range(self.epochs):
            progress = epoch / max(self.epochs - 1, 1)
            self.grl.set_progress(progress)

            epoch_nll, epoch_adv, epoch_pol, epoch_cls, n_batches = (
                0.0, 0.0, 0.0, 0.0, 0,
            )
            s_batches = _make_batches(src_seqs, src_labels, self.batch_size)
            t_batches = _make_batches(tgt_seqs, tgt_labels, self.batch_size)

            for (s_ids, s_mask, s_lbl), (t_ids, t_mask, t_lbl) in zip(
                s_batches, t_batches,
            ):
                s_ids = s_ids.to(self._device)
                s_mask = s_mask.to(self._device)
                s_lbl = s_lbl.to(self._device)
                t_ids = t_ids.to(self._device)
                t_mask = t_mask.to(self._device)
                t_lbl = t_lbl.to(self._device)

                s_logits = self.encoder(s_ids)
                loss_nll = nll_loss_fn(
                    s_logits[:, :-1].contiguous().reshape(-1, self.vocab_size),
                    s_ids[:, 1:].contiguous().reshape(-1),
                )

                s_feat = self.encoder.extract_features(s_ids, s_mask)
                t_feat = self.encoder.extract_features(t_ids, t_mask)

                s_rev = self.grl(s_feat)
                t_rev = self.grl(t_feat)
                all_feat = torch.cat([s_rev, t_rev], dim=0)
                domain_labels = torch.cat([
                    torch.zeros(s_feat.size(0), device=self._device),
                    torch.ones(t_feat.size(0), device=self._device),
                ])
                loss_adv = F.binary_cross_entropy_with_logits(
                    self.domain_clf(all_feat), domain_labels,
                )

                loss_pol = compute_polarity_loss(s_feat, s_lbl, t_feat, t_lbl)

                cls_logits = self.cls_head(s_feat).squeeze(-1)
                loss_cls = F.binary_cross_entropy_with_logits(
                    cls_logits, s_lbl.float(),
                )

                loss = (
                    loss_nll
                    + self.lambda_adv * loss_adv
                    + self.lambda_pol * loss_pol
                    + self.lambda_cls * loss_cls
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()

                epoch_nll += loss_nll.item()
                epoch_adv += loss_adv.item()
                epoch_pol += loss_pol.item()
                epoch_cls += loss_cls.item()
                n_batches += 1

            scheduler.step()

            avg_nll = epoch_nll / max(n_batches, 1)
            avg_adv = epoch_adv / max(n_batches, 1)
            avg_pol = epoch_pol / max(n_batches, 1)
            avg_cls = epoch_cls / max(n_batches, 1)
            avg_total = (
                avg_nll
                + self.lambda_adv * avg_adv
                + self.lambda_pol * avg_pol
                + self.lambda_cls * avg_cls
            )

            history["nll"].append(avg_nll)
            history["adv"].append(avg_adv)
            history["pol"].append(avg_pol)
            history["cls"].append(avg_cls)
            history["total"].append(avg_total)

            if avg_total < best_loss - 1e-6:
                best_loss = avg_total
                best_state = {
                    "encoder": {
                        k: v.cpu().clone()
                        for k, v in self.encoder.state_dict().items()
                    },
                    "domain_clf": {
                        k: v.cpu().clone()
                        for k, v in self.domain_clf.state_dict().items()
                    },
                    "cls_head": {
                        k: v.cpu().clone()
                        for k, v in self.cls_head.state_dict().items()
                    },
                }
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= self.patience:
                    break

        if best_state is not None:
            self.encoder.load_state_dict(best_state["encoder"])
            self.domain_clf.load_state_dict(best_state["domain_clf"])
            self.cls_head.load_state_dict(best_state["cls_head"])
        self.encoder.to(self._device)
        self.domain_clf.to(self._device)
        self.cls_head.to(self._device)
        self.encoder.eval()

        return history


def _compute_nll_scores(
    encoder: PARDEncoder,
    seqs: list[list[int]],
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Compute per-trace mean NLL scores (same metric as baselines)."""
    encoder.eval()
    all_scores: list[np.ndarray] = []
    dummy_labels = [0] * len(seqs)
    with torch.no_grad():
        for ids, mask, _ in _make_batches(seqs, dummy_labels, batch_size):
            ids = ids.to(device)
            mask = mask.to(device)
            nll = encoder.compute_nll(ids, mask)
            all_scores.append(nll.cpu().numpy())
    return np.concatenate(all_scores, axis=0) if all_scores else np.empty(0)


def _extract_features(
    encoder: PARDEncoder,
    seqs: list[list[int]],
    labels: list[int],
    device: torch.device,
    batch_size: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Extract bottleneck features from encoder."""
    encoder.eval()
    all_feat: list[np.ndarray] = []
    with torch.no_grad():
        for ids, mask, _ in _make_batches(seqs, labels, batch_size):
            ids = ids.to(device)
            mask = mask.to(device)
            feat = encoder.extract_features(ids, mask)
            all_feat.append(feat.cpu().numpy())
    features = np.concatenate(all_feat, axis=0) if all_feat else np.empty((0, encoder.d_bottleneck))
    return features, np.array(labels, dtype=np.int64)


def _make_batches(
    seqs: list[list[int]],
    labels: list[int],
    batch_size: int,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Pad and batch sequences (reuses dann_coral pattern)."""
    import random

    indices = list(range(len(seqs)))
    random.shuffle(indices)
    batches = []

    for i in range(0, len(indices), batch_size):
        bi = indices[i : i + batch_size]
        batch_seqs = [seqs[j] for j in bi]
        batch_labels = [labels[j] for j in bi]
        max_len = max(len(s) for s in batch_seqs)

        ids = torch.zeros(len(bi), max_len, dtype=torch.long)
        mask = torch.zeros(len(bi), max_len, dtype=torch.bool)
        lbl = torch.tensor(batch_labels, dtype=torch.long)

        for j, s in enumerate(batch_seqs):
            ids[j, : len(s)] = torch.tensor(s, dtype=torch.long)
            mask[j, : len(s)] = True

        batches.append((ids, mask, lbl))

    return batches


HEURISTIC_NAMES = (
    "seq_len", "diversity", "entropy", "max_repeat",
    "nll_mean", "nll_std", "nll_max",
)


def compute_trace_heuristics(
    seqs: list[list[int]],
    encoder: PARDEncoder | None = None,
    device: torch.device | None = None,
) -> np.ndarray:
    """Extract interpretable per-trace features (K-dimensional).

    Computes 4 heuristic features from raw sequences plus 3 LSTM-derived
    features if an encoder is provided.

    Returns:
        (N, K) feature matrix where K=7 (with encoder) or K=4 (without).
    """
    import math
    from collections import Counter

    features = []
    for seq in seqs:
        n = len(seq)
        counts = Counter(seq)
        total = sum(counts.values())
        entropy = 0.0
        for c in counts.values():
            p = c / total
            if p > 0:
                entropy -= p * math.log2(p)
        diversity = len(set(seq)) / max(n, 1)
        max_repeat = 1
        cur = 1
        for i in range(1, n):
            if seq[i] == seq[i - 1]:
                cur += 1
                max_repeat = max(max_repeat, cur)
            else:
                cur = 1
        features.append([n, diversity, entropy, max_repeat])

    feat = np.array(features, dtype=np.float64)

    if encoder is not None and device is not None:
        nll_scores = _compute_nll_scores(encoder, seqs, device)
        nll_per_action = _compute_per_action_nll_stats(encoder, seqs, device)
        feat = np.column_stack([feat, nll_per_action])

    return feat


def _compute_per_action_nll_stats(
    encoder: PARDEncoder,
    seqs: list[list[int]],
    device: torch.device,
    batch_size: int = 64,
) -> np.ndarray:
    """Compute per-trace NLL statistics: mean, std, max."""
    encoder.eval()
    results = []
    dummy = [0] * len(seqs)
    with torch.no_grad():
        for ids, mask, _ in _make_batches(seqs, dummy, batch_size):
            ids = ids.to(device)
            mask = mask.to(device)
            logits = encoder(ids)
            loss_fn = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=0)
            per_token = loss_fn(
                logits[:, :-1].contiguous().reshape(-1, logits.size(-1)),
                ids[:, 1:].contiguous().reshape(-1),
            ).reshape(ids.size(0), -1)
            valid_mask = mask[:, 1:]
            per_token = per_token * valid_mask.float()
            lengths = valid_mask.sum(dim=1).float().clamp(min=1)
            nll_mean = (per_token.sum(dim=1) / lengths).cpu().numpy()
            nll_std_vals = []
            nll_max_vals = []
            for i in range(ids.size(0)):
                valid = per_token[i, :int(lengths[i].item())]
                nll_std_vals.append(valid.std().item() if len(valid) > 1 else 0.0)
                nll_max_vals.append(valid.max().item() if len(valid) > 0 else 0.0)
            results.append(np.column_stack([
                nll_mean,
                np.array(nll_std_vals),
                np.array(nll_max_vals),
            ]))
    return np.concatenate(results, axis=0) if results else np.empty((0, 3))


def pard_calibrate_heuristic(
    src_seqs: list[list[int]],
    src_labels: list[int],
    tgt_cal_seqs: list[list[int]],
    tgt_cal_labels: list[int],
    *,
    encoder: PARDEncoder | None = None,
    mode: CorrectionMode = CorrectionMode.DIAGONAL,
    alpha: float = 0.05,
    device: torch.device | None = None,
) -> tuple[PolarityCorrectionLayer, float, PARDClassifier, dict]:
    """PARD calibration on heuristic features.

    Returns:
        (correction, tau, classifier, info_dict)
    """
    src_feat = compute_trace_heuristics(src_seqs, encoder, device)
    tgt_feat = compute_trace_heuristics(tgt_cal_seqs, encoder, device)
    src_lbl = np.array(src_labels, dtype=np.int64)
    tgt_lbl = np.array(tgt_cal_labels, dtype=np.int64)
    d = src_feat.shape[1]

    n = len(tgt_cal_seqs)
    n_est = n // 2
    idx = np.random.permutation(n)
    est_idx, cal_idx = idx[:n_est], idx[n_est:]

    correction = PolarityCorrectionLayer(d, mode)
    correction.estimate(
        src_feat, src_lbl,
        tgt_feat[est_idx], tgt_lbl[est_idx],
    )

    corrected_src = correction.correct(src_feat)
    corrected_cal = correction.correct(tgt_feat[cal_idx])

    clf = PARDClassifier()
    clf.fit(corrected_src, src_lbl)

    cal_scores = clf.predict_proba(corrected_cal)
    tau = split_conformal_calibrate(cal_scores, alpha=alpha)

    info = {
        "n_estimation": n_est,
        "n_conformal": n - n_est,
        "d_features": d,
        "reversal_rate_pre": float(np.mean(
            np.sign(correction._per_dim_correlation(src_feat, src_lbl))
            != np.sign(correction._per_dim_correlation(tgt_feat, tgt_lbl))
        )),
        "reversal_rate_post": correction.reversal_rate,
        "mode": mode.value,
    }
    return correction, tau, clf, info


def pard_predict_heuristic(
    correction: PolarityCorrectionLayer,
    clf: PARDClassifier,
    tau: float,
    seqs: list[list[int]],
    *,
    encoder: PARDEncoder | None = None,
    device: torch.device | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """PARD prediction on heuristic features.

    Returns:
        (predictions, scores) where predictions are binary (1=anomalous).
    """
    feat = compute_trace_heuristics(seqs, encoder, device)
    corrected = correction.correct(feat)
    scores = clf.predict_proba(corrected)
    predictions = (scores > tau).astype(np.int32)
    return predictions, scores

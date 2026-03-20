"""DANN and CORAL domain adaptation baselines.

These are standard DA methods that align marginal distributions
P(z|source) ~ P(z|target) but do NOT address polarity reversal
(sign(corr(z_k, y)) may differ across domains).

Hypothesis: both will improve cross-domain feature alignment but
fail to fix polarity reversal, resulting in AUROC near or below 0.5
on reversed domain pairs.

References:
    DANN: Ganin et al., "Domain-Adversarial Training of Neural Networks"
          (JMLR 2016)
    CORAL: Sun & Saenko, "Deep CORAL: Correlation Alignment for Deep
           Domain Adaptation" (ECCV 2016 Workshops)
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from baselines.all_baselines import IndependentScorerBaseline
from trustlens.dapl import DomainClassifier, GradientReversalLayer, LSTMEncoder


class DANNBaseline(IndependentScorerBaseline):
    """DANN: LSTM + GRL + domain classifier (no polarity term).

    Aligns marginal feature distributions via adversarial training
    but does not address feature-label correlation direction.
    """

    def __init__(
        self,
        d_embed: int = 128,
        d_hidden: int = 128,
        lambda_adv: float = 0.1,
        lr: float = 1e-3,
        epochs: int = 50,
        patience: int = 5,
        batch_size: int = 64,
    ) -> None:
        self._d_embed = d_embed
        self._d_hidden = d_hidden
        self._lambda_adv = lambda_adv
        self._lr = lr
        self._epochs = epochs
        self._patience = patience
        self._batch_size = batch_size
        self._encoder: LSTMEncoder | None = None
        self._grl: GradientReversalLayer | None = None
        self._domain_clf: DomainClassifier | None = None
        self._threshold: float = 0.0
        self._device = torch.device("cpu")

    def name(self) -> str:
        return "DANN"

    def fit_cross_domain(
        self,
        vocab_size: int,
        src_seqs: list[list[int]],
        src_labels: list[int],
        tgt_seqs: list[list[int]],
        tgt_labels: list[int],
        cal_seqs: list[list[int]],
    ) -> None:
        """Train DANN and set threshold from calibration scores."""
        from torch.optim.lr_scheduler import CosineAnnealingLR

        if torch.backends.mps.is_available():
            self._device = torch.device("mps")
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")

        self._encoder = LSTMEncoder(vocab_size, self._d_embed, self._d_hidden)
        self._grl = GradientReversalLayer(lambda_max=1.0)
        self._domain_clf = DomainClassifier(self._d_hidden)
        self._encoder.to(self._device)
        self._domain_clf.to(self._device)

        params = list(self._encoder.parameters()) + list(self._domain_clf.parameters())
        optimizer = torch.optim.AdamW(params, lr=self._lr, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=self._epochs)
        nll_loss_fn = nn.CrossEntropyLoss(ignore_index=0)

        best_loss = float("inf")
        best_state = None
        patience_ctr = 0

        self._encoder.train()
        self._domain_clf.train()

        for epoch in range(self._epochs):
            progress = epoch / max(self._epochs - 1, 1)
            self._grl.set_progress(progress)

            epoch_loss, n_batches = 0.0, 0
            s_batches = _make_batches(src_seqs, src_labels, self._batch_size)
            t_batches = _make_batches(tgt_seqs, tgt_labels, self._batch_size)

            for (s_ids, s_mask, _), (t_ids, t_mask, _) in zip(s_batches, t_batches):
                s_ids = s_ids.to(self._device)
                s_mask = s_mask.to(self._device)
                t_ids = t_ids.to(self._device)
                t_mask = t_mask.to(self._device)

                s_logits = self._encoder(s_ids)
                loss_nll = nll_loss_fn(
                    s_logits[:, :-1].contiguous().reshape(-1, vocab_size),
                    s_ids[:, 1:].contiguous().reshape(-1),
                )

                s_feat = self._encoder.trace_features(s_ids, s_mask)
                t_feat = self._encoder.trace_features(t_ids, t_mask)
                s_rev = self._grl(s_feat)
                t_rev = self._grl(t_feat)
                all_feat = torch.cat([s_rev, t_rev], dim=0)
                domain_lbl = torch.cat([
                    torch.zeros(s_feat.size(0), device=self._device),
                    torch.ones(t_feat.size(0), device=self._device),
                ])
                loss_adv = F.binary_cross_entropy_with_logits(
                    self._domain_clf(all_feat), domain_lbl,
                )

                loss = loss_nll + self._lambda_adv * loss_adv
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg = epoch_loss / max(n_batches, 1)
            if avg < best_loss - 1e-6:
                best_loss = avg
                best_state = {
                    k: v.cpu().clone() for k, v in self._encoder.state_dict().items()
                }
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= self._patience:
                    break

        if best_state is not None:
            self._encoder.load_state_dict(best_state)
        self._encoder.to(self._device)
        self._encoder.eval()

        cal_scores = self._score_seqs(cal_seqs)
        self._threshold = float(np.percentile(cal_scores, 95))

    def _score_seqs(self, seqs: list[list[int]]) -> np.ndarray:
        self._encoder.eval()
        scores: list[float] = []
        with torch.no_grad():
            for ids, mask, _ in _make_batches(seqs, [0] * len(seqs), self._batch_size):
                ids = ids.to(self._device)
                mask = mask.to(self._device)
                nll = self._encoder.compute_nll(ids, mask)
                scores.extend(nll.cpu().tolist())
        return np.array(scores, dtype=np.float64)

    def fit_traces(self, train_traces, cal_traces, vocab) -> None:
        pass

    def score_traces(self, traces, vocab) -> np.ndarray:
        seqs = [[vocab.encode_tool(a.tool) for a in t.actions] for t in traces]
        return self._score_seqs(seqs)

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return (scores > self._threshold).astype(np.int32)


def _coral_loss(
    src_features: torch.Tensor,
    tgt_features: torch.Tensor,
) -> torch.Tensor:
    """CORAL loss: Frobenius norm of covariance difference.

    Args:
        src_features: (N_s, D) source features.
        tgt_features: (N_t, D) target features.

    Returns:
        Scalar CORAL loss.
    """
    d = src_features.size(1)

    src_mean = src_features.mean(0, keepdim=True)
    tgt_mean = tgt_features.mean(0, keepdim=True)
    src_centered = src_features - src_mean
    tgt_centered = tgt_features - tgt_mean

    n_s = max(src_features.size(0) - 1, 1)
    n_t = max(tgt_features.size(0) - 1, 1)
    cov_src = (src_centered.T @ src_centered) / n_s
    cov_tgt = (tgt_centered.T @ tgt_centered) / n_t

    diff = cov_src - cov_tgt
    return (diff * diff).sum() / (4 * d * d)


class CORALBaseline(IndependentScorerBaseline):
    """Deep CORAL: LSTM + correlation alignment loss.

    Minimizes Frobenius norm of source/target covariance matrix
    difference. Aligns second-order statistics but does not
    address polarity (sign of feature-label correlations).
    """

    def __init__(
        self,
        d_embed: int = 128,
        d_hidden: int = 128,
        lambda_coral: float = 0.5,
        lr: float = 1e-3,
        epochs: int = 50,
        patience: int = 5,
        batch_size: int = 64,
    ) -> None:
        self._d_embed = d_embed
        self._d_hidden = d_hidden
        self._lambda_coral = lambda_coral
        self._lr = lr
        self._epochs = epochs
        self._patience = patience
        self._batch_size = batch_size
        self._encoder: LSTMEncoder | None = None
        self._threshold: float = 0.0
        self._device = torch.device("cpu")

    def name(self) -> str:
        return "CORAL"

    def fit_cross_domain(
        self,
        vocab_size: int,
        src_seqs: list[list[int]],
        src_labels: list[int],
        tgt_seqs: list[list[int]],
        tgt_labels: list[int],
        cal_seqs: list[list[int]],
    ) -> None:
        """Train CORAL baseline and set threshold."""
        from torch.optim.lr_scheduler import CosineAnnealingLR

        if torch.backends.mps.is_available():
            self._device = torch.device("mps")
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")

        self._encoder = LSTMEncoder(vocab_size, self._d_embed, self._d_hidden)
        self._encoder.to(self._device)

        optimizer = torch.optim.AdamW(
            self._encoder.parameters(), lr=self._lr, weight_decay=0.01,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self._epochs)
        nll_loss_fn = nn.CrossEntropyLoss(ignore_index=0)

        best_loss = float("inf")
        best_state = None
        patience_ctr = 0

        self._encoder.train()
        for epoch in range(self._epochs):
            epoch_loss, n_batches = 0.0, 0
            s_batches = _make_batches(src_seqs, src_labels, self._batch_size)
            t_batches = _make_batches(tgt_seqs, tgt_labels, self._batch_size)

            for (s_ids, s_mask, _), (t_ids, t_mask, _) in zip(s_batches, t_batches):
                s_ids = s_ids.to(self._device)
                s_mask = s_mask.to(self._device)
                t_ids = t_ids.to(self._device)
                t_mask = t_mask.to(self._device)

                s_logits = self._encoder(s_ids)
                loss_nll = nll_loss_fn(
                    s_logits[:, :-1].contiguous().reshape(-1, vocab_size),
                    s_ids[:, 1:].contiguous().reshape(-1),
                )

                s_feat = self._encoder.trace_features(s_ids, s_mask)
                t_feat = self._encoder.trace_features(t_ids, t_mask)
                loss_coral = _coral_loss(s_feat, t_feat)

                loss = loss_nll + self._lambda_coral * loss_coral
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self._encoder.parameters(), 1.0)
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            avg = epoch_loss / max(n_batches, 1)
            if avg < best_loss - 1e-6:
                best_loss = avg
                best_state = {
                    k: v.cpu().clone() for k, v in self._encoder.state_dict().items()
                }
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= self._patience:
                    break

        if best_state is not None:
            self._encoder.load_state_dict(best_state)
        self._encoder.to(self._device)
        self._encoder.eval()

        cal_scores = self._score_seqs(cal_seqs)
        self._threshold = float(np.percentile(cal_scores, 95))

    def _score_seqs(self, seqs: list[list[int]]) -> np.ndarray:
        self._encoder.eval()
        scores: list[float] = []
        with torch.no_grad():
            for ids, mask, _ in _make_batches(seqs, [0] * len(seqs), self._batch_size):
                ids = ids.to(self._device)
                mask = mask.to(self._device)
                nll = self._encoder.compute_nll(ids, mask)
                scores.extend(nll.cpu().tolist())
        return np.array(scores, dtype=np.float64)

    def fit_traces(self, train_traces, cal_traces, vocab) -> None:
        pass

    def score_traces(self, traces, vocab) -> np.ndarray:
        seqs = [[vocab.encode_tool(a.tool) for a in t.actions] for t in traces]
        return self._score_seqs(seqs)

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return (scores > self._threshold).astype(np.int32)


def _make_batches(
    seqs: list[list[int]],
    labels: list[int],
    batch_size: int,
) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Pad and batch sequences (shared utility)."""
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


def get_da_baselines() -> list[IndependentScorerBaseline]:
    """Return DANN and CORAL baselines for comparison."""
    return [DANNBaseline(), CORALBaseline()]

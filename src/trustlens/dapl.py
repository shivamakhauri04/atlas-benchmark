"""Domain-Adversarial Polarity Loss (DAPL).

Upgrades PAFC's manual sign-flip to a learned polarity-aware loss.
Architecture: shared LSTM encoder -> GRL -> domain classifier +
polarity alignment regularizer.

Loss: L = L_ad + lambda_adv * L_adversarial + lambda_pol * L_polarity
  - L_ad: anomaly detection loss (NLL from next-token prediction)
  - L_adversarial: GRL + domain classifier (standard DANN component)
  - L_polarity: cosine similarity penalty forcing feature-label
    correlation signs to align across domains

PAFC is a special case: DAPL with lambda_adv=0 and hard sign-flip
instead of soft alignment.

Key novelty: standard DA (DANN) aligns marginal P(z); DAPL aligns
the feature-label *direction* sign(corr(z_k, y)), which is the
actual failure mode in cross-domain anomaly detection.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function


class GradientReversalFn(Function):
    """Gradient Reversal Layer (Ganin & Lempitsky, 2015)."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx.lambda_ = lambda_
        return x.clone()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambda_ * grad_output, None


class GradientReversalLayer(nn.Module):
    """GRL with progressive lambda scheduling."""

    def __init__(self, lambda_max: float = 1.0) -> None:
        super().__init__()
        self.lambda_max = lambda_max
        self._progress: float = 0.0

    def set_progress(self, progress: float) -> None:
        self._progress = min(max(progress, 0.0), 1.0)

    @property
    def current_lambda(self) -> float:
        import math

        p = self._progress
        return self.lambda_max * (2.0 / (1.0 + math.exp(-10.0 * p)) - 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return GradientReversalFn.apply(x, self.current_lambda)


class DomainClassifier(nn.Module):
    """2-layer MLP for domain discrimination."""

    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def compute_polarity_loss(
    features_src: torch.Tensor,
    labels_src: torch.Tensor,
    features_tgt: torch.Tensor,
    labels_tgt: torch.Tensor,
) -> torch.Tensor:
    """Polarity alignment loss.

    Computes per-dimension correlation signs between features and
    labels in source and target, then penalizes misalignment.

    L_polarity = -mean_k(sign(corr_src_k) * d_tgt_k)

    where d_tgt_k is the soft Cohen's d proxy (mean difference
    between label=1 and label=0 groups) for dimension k in target.

    Args:
        features_src: (N_s, D) source features.
        labels_src: (N_s,) binary source labels.
        features_tgt: (N_t, D) target features.
        labels_tgt: (N_t,) binary target labels.

    Returns:
        Scalar polarity loss (lower = better alignment).
    """
    src_pos = labels_src == 1
    src_neg = labels_src == 0
    tgt_pos = labels_tgt == 1
    tgt_neg = labels_tgt == 0

    if src_pos.sum() < 2 or src_neg.sum() < 2:
        return torch.tensor(0.0, device=features_src.device)
    if tgt_pos.sum() < 2 or tgt_neg.sum() < 2:
        return torch.tensor(0.0, device=features_src.device)

    src_d = features_src[src_pos].mean(0) - features_src[src_neg].mean(0)
    tgt_d = features_tgt[tgt_pos].mean(0) - features_tgt[tgt_neg].mean(0)

    src_sign = src_d.sign()
    alignment = src_sign * tgt_d
    return -alignment.mean()


class LSTMEncoder(nn.Module):
    """LSTM backbone shared by DAPL, DANN, and CORAL."""

    def __init__(
        self,
        vocab_size: int,
        d_embed: int = 128,
        d_hidden: int = 128,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_embed, padding_idx=0)
        self.lstm = nn.LSTM(d_embed, d_hidden, batch_first=True)
        self.head = nn.Linear(d_hidden, vocab_size)
        self.d_hidden = d_hidden

    def encode(self, tool_ids: torch.Tensor) -> torch.Tensor:
        """Return per-step hidden states (B, T, d_hidden)."""
        emb = self.embed(tool_ids)
        out, _ = self.lstm(emb)
        return out

    def forward(self, tool_ids: torch.Tensor) -> torch.Tensor:
        """Return next-tool prediction logits (B, T, vocab_size)."""
        return self.head(self.encode(tool_ids))

    def trace_features(self, tool_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Mean-pool hidden states over valid tokens -> (B, d_hidden)."""
        h = self.encode(tool_ids)
        lengths = mask.float().sum(dim=1, keepdim=True).clamp(min=1)
        return (h * mask.unsqueeze(-1).float()).sum(dim=1) / lengths

    def compute_nll(self, tool_ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Per-trace mean NLL anomaly score (B,)."""
        logits = self.forward(tool_ids)
        log_probs = F.log_softmax(logits, dim=-1)
        targets = tool_ids[:, 1:]
        preds = log_probs[:, :-1]
        nll = -preds.gather(2, targets.unsqueeze(-1)).squeeze(-1)
        nll_full = torch.zeros_like(tool_ids, dtype=torch.float)
        nll_full[:, 1:] = nll
        nll_full = nll_full * mask.float()
        lengths = mask.float().sum(dim=1).clamp(min=1)
        return nll_full.sum(dim=1) / lengths


class DAPLTrainer:
    """Trains an LSTM encoder with DAPL objective.

    Loss = L_nll + lambda_adv * L_adversarial + lambda_pol * L_polarity

    Args:
        vocab_size: Tool vocabulary size.
        d_embed: Embedding dimension.
        d_hidden: LSTM hidden dimension.
        lambda_adv: Weight for adversarial (domain classifier) loss.
        lambda_pol: Weight for polarity alignment loss.
        lr: Learning rate.
        epochs: Training epochs.
        patience: Early stopping patience.
        batch_size: Batch size.
    """

    def __init__(
        self,
        vocab_size: int,
        d_embed: int = 128,
        d_hidden: int = 128,
        lambda_adv: float = 0.1,
        lambda_pol: float = 0.5,
        lr: float = 1e-3,
        epochs: int = 50,
        patience: int = 5,
        batch_size: int = 64,
    ) -> None:
        self.vocab_size = vocab_size
        self.lambda_adv = lambda_adv
        self.lambda_pol = lambda_pol
        self.lr = lr
        self.epochs = epochs
        self.patience = patience
        self.batch_size = batch_size

        self.encoder = LSTMEncoder(vocab_size, d_embed, d_hidden)
        self.grl = GradientReversalLayer(lambda_max=1.0)
        self.domain_clf = DomainClassifier(d_hidden)

        self._device = torch.device("cpu")

    def to(self, device: torch.device) -> "DAPLTrainer":
        self._device = device
        self.encoder.to(device)
        self.domain_clf.to(device)
        return self

    def train(
        self,
        src_seqs: list[list[int]],
        src_labels: list[int],
        tgt_seqs: list[list[int]],
        tgt_labels: list[int],
    ) -> dict[str, list[float]]:
        """Train encoder with DAPL loss on source + target data.

        Args:
            src_seqs: Source domain tool-ID sequences.
            src_labels: Source binary labels (0/1).
            tgt_seqs: Target domain tool-ID sequences.
            tgt_labels: Target binary labels (0/1).

        Returns:
            Training history with per-epoch losses.
        """
        from torch.optim.lr_scheduler import CosineAnnealingLR

        params = (
            list(self.encoder.parameters())
            + list(self.domain_clf.parameters())
        )
        optimizer = torch.optim.AdamW(params, lr=self.lr, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=self.epochs)
        nll_loss_fn = nn.CrossEntropyLoss(ignore_index=0)

        history: dict[str, list[float]] = {
            "nll": [], "adv": [], "pol": [], "total": [],
        }
        best_loss = float("inf")
        best_state = None
        patience_ctr = 0

        self.encoder.train()
        self.domain_clf.train()

        for epoch in range(self.epochs):
            progress = epoch / max(self.epochs - 1, 1)
            self.grl.set_progress(progress)

            epoch_nll, epoch_adv, epoch_pol, n_batches = 0.0, 0.0, 0.0, 0

            src_batches = self._make_batches(src_seqs, src_labels)
            tgt_batches = self._make_batches(tgt_seqs, tgt_labels)

            for (s_ids, s_mask, s_lbl), (t_ids, t_mask, t_lbl) in zip(
                src_batches, tgt_batches
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

                s_feat = self.encoder.trace_features(s_ids, s_mask)
                t_feat = self.encoder.trace_features(t_ids, t_mask)

                s_feat_rev = self.grl(s_feat)
                t_feat_rev = self.grl(t_feat)
                all_feat = torch.cat([s_feat_rev, t_feat_rev], dim=0)
                domain_labels = torch.cat([
                    torch.zeros(s_feat.size(0), device=self._device),
                    torch.ones(t_feat.size(0), device=self._device),
                ])
                domain_logits = self.domain_clf(all_feat)
                loss_adv = F.binary_cross_entropy_with_logits(
                    domain_logits, domain_labels,
                )

                loss_pol = compute_polarity_loss(s_feat, s_lbl, t_feat, t_lbl)

                loss = (
                    loss_nll
                    + self.lambda_adv * loss_adv
                    + self.lambda_pol * loss_pol
                )

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()

                epoch_nll += loss_nll.item()
                epoch_adv += loss_adv.item()
                epoch_pol += loss_pol.item()
                n_batches += 1

            scheduler.step()

            avg_nll = epoch_nll / max(n_batches, 1)
            avg_adv = epoch_adv / max(n_batches, 1)
            avg_pol = epoch_pol / max(n_batches, 1)
            avg_total = avg_nll + self.lambda_adv * avg_adv + self.lambda_pol * avg_pol

            history["nll"].append(avg_nll)
            history["adv"].append(avg_adv)
            history["pol"].append(avg_pol)
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
                }
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= self.patience:
                    break

        if best_state is not None:
            self.encoder.load_state_dict(best_state["encoder"])
            self.domain_clf.load_state_dict(best_state["domain_clf"])
        self.encoder.to(self._device)
        self.domain_clf.to(self._device)
        self.encoder.eval()

        return history

    def score_traces(
        self,
        seqs: list[list[int]],
    ) -> list[float]:
        """Score traces by mean NLL (higher = more anomalous)."""
        self.encoder.eval()
        scores: list[float] = []
        with torch.no_grad():
            for batch_ids, batch_mask, _ in self._make_batches(
                seqs, [0] * len(seqs),
            ):
                batch_ids = batch_ids.to(self._device)
                batch_mask = batch_mask.to(self._device)
                nll = self.encoder.compute_nll(batch_ids, batch_mask)
                scores.extend(nll.cpu().tolist())
        return scores

    def _make_batches(
        self,
        seqs: list[list[int]],
        labels: list[int],
    ) -> list[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """Pad and batch sequences."""
        import random

        indices = list(range(len(seqs)))
        random.shuffle(indices)
        batches = []

        for i in range(0, len(indices), self.batch_size):
            bi = indices[i : i + self.batch_size]
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

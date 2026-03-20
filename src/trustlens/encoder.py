"""Transformer trace encoder.

Encodes variable-length agent action traces into fixed-dimensional
embeddings for downstream conformal anomaly detection.

Architecture:
    - Tool embedding + optional args/outcome hash embeddings (learned)
    - Positional encoding (sinusoidal)
    - N transformer encoder layers (causal masking)
    - Per-step NLL: predict next tool from current prefix -> anomaly score
    - (Consequence-aware): additionally predict next outcome -> joint score

The NLL at each timestep t measures how "surprising" action_t is given
actions_{0..t-1}. High NLL = anomalous behavior.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512) -> None:
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, : x.size(1)]


class TransformerTraceEncoder(nn.Module):
    """Causal transformer for agent trace anomaly detection via NLL.

    Input: (tool_ids, args_hashes, mask) per trace.
    Output: per-step NLL scores (higher = more anomalous).

    When use_outcome=True, additionally predicts the next outcome hash,
    enabling consequence-aware scoring where the joint anomaly score
    combines tool prediction NLL with outcome prediction NLL.
    """

    def __init__(
        self,
        vocab_size: int,
        d_model: int = 128,
        n_heads: int = 4,
        n_layers: int = 3,
        d_ff: int = 512,
        dropout: float = 0.1,
        args_buckets: int = 2**16,
        max_len: int = 512,
        use_args: bool = True,
        use_outcome: bool = False,
        outcome_buckets: int = 2**16,
        outcome_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.use_args = use_args
        self.use_outcome = use_outcome
        self.outcome_weight = outcome_weight

        self.tool_embed = nn.Embedding(vocab_size, d_model, padding_idx=0)

        input_dim = d_model
        if use_args:
            self.args_embed = nn.Embedding(args_buckets, d_model // 4)
            input_dim += d_model // 4
        if use_outcome:
            self.outcome_embed = nn.Embedding(outcome_buckets, d_model // 4)
            input_dim += d_model // 4

        if input_dim != d_model:
            self.input_proj = nn.Linear(input_dim, d_model)
        else:
            self.input_proj = None

        self.pos_enc = SinusoidalPositionalEncoding(d_model, max_len)
        self.input_dropout = nn.Dropout(dropout)

        if n_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=n_heads,
                dim_feedforward=d_ff,
                dropout=dropout,
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        else:
            self.transformer = None

        self.head = nn.Linear(d_model, vocab_size)

        if use_outcome:
            self.outcome_head = nn.Linear(d_model, outcome_buckets)

    def _causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Upper-triangular causal mask (True = blocked)."""
        return torch.triu(
            torch.ones(seq_len, seq_len, device=device, dtype=torch.bool),
            diagonal=1,
        )

    def _encode(
        self,
        tool_ids: torch.Tensor,
        args_hashes: torch.Tensor,
        mask: torch.Tensor,
        outcome_hashes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Shared encoder backbone returning hidden states (B, T, d_model)."""
        parts = [self.tool_embed(tool_ids)]
        if self.use_args:
            parts.append(self.args_embed(args_hashes))
        if self.use_outcome and outcome_hashes is not None:
            parts.append(self.outcome_embed(outcome_hashes))

        if len(parts) > 1:
            x = torch.cat(parts, dim=-1)
        else:
            x = parts[0]

        if self.input_proj is not None:
            x = self.input_proj(x)

        x = self.pos_enc(x)
        x = self.input_dropout(x)

        seq_len = x.size(1)
        causal = self._causal_mask(seq_len, x.device)
        pad_mask = ~mask

        if self.transformer is not None:
            x = self.transformer(x, mask=causal, src_key_padding_mask=pad_mask)
        return x

    def forward(
        self,
        tool_ids: torch.Tensor,
        args_hashes: torch.Tensor,
        mask: torch.Tensor,
        outcome_hashes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass returning per-step tool logits.

        Args:
            tool_ids: (B, T) tool integer IDs.
            args_hashes: (B, T) hashed argument buckets.
            mask: (B, T) bool, True for real tokens.
            outcome_hashes: (B, T) hashed outcome buckets (optional).

        Returns:
            logits: (B, T, vocab_size) next-tool prediction logits.
        """
        h = self._encode(tool_ids, args_hashes, mask, outcome_hashes)
        return self.head(h)

    def forward_joint(
        self,
        tool_ids: torch.Tensor,
        args_hashes: torch.Tensor,
        mask: torch.Tensor,
        outcome_hashes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both tool and outcome logits.

        Returns:
            (tool_logits, outcome_logits) each of shape (B, T, vocab/buckets).
        """
        h = self._encode(tool_ids, args_hashes, mask, outcome_hashes)
        tool_logits = self.head(h)
        outcome_logits = self.outcome_head(h)
        return tool_logits, outcome_logits

    def compute_nll(
        self,
        tool_ids: torch.Tensor,
        args_hashes: torch.Tensor,
        mask: torch.Tensor,
        outcome_hashes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute per-step negative log-likelihood (anomaly score).

        For each position t, the NLL is computed as:
            -log P(tool_ids[t] | tool_ids[0..t-1])

        Position 0 gets NLL=0 (no prior context to judge).

        Returns:
            nll: (B, T) per-step NLL scores. Masked positions are 0.
        """
        logits = self.forward(tool_ids, args_hashes, mask, outcome_hashes)
        log_probs = F.log_softmax(logits, dim=-1)

        targets = tool_ids[:, 1:]
        preds = log_probs[:, :-1]

        nll = -preds.gather(2, targets.unsqueeze(-1)).squeeze(-1)

        nll_full = torch.zeros_like(tool_ids, dtype=torch.float)
        nll_full[:, 1:] = nll
        nll_full = nll_full * mask.float()
        return nll_full

    def compute_joint_nll(
        self,
        tool_ids: torch.Tensor,
        args_hashes: torch.Tensor,
        mask: torch.Tensor,
        outcome_hashes: torch.Tensor,
    ) -> torch.Tensor:
        """Compute joint NLL: tool prediction + outcome prediction.

        Score = NLL(next_tool) + outcome_weight * NLL(next_outcome)

        Returns:
            joint_nll: (B, T) per-step joint NLL scores.
        """
        tool_logits, outcome_logits = self.forward_joint(
            tool_ids,
            args_hashes,
            mask,
            outcome_hashes,
        )

        tool_log_probs = F.log_softmax(tool_logits, dim=-1)
        tool_targets = tool_ids[:, 1:]
        tool_preds = tool_log_probs[:, :-1]
        tool_nll = -tool_preds.gather(2, tool_targets.unsqueeze(-1)).squeeze(-1)

        outcome_log_probs = F.log_softmax(outcome_logits, dim=-1)
        outcome_targets = outcome_hashes[:, 1:]
        outcome_preds = outcome_log_probs[:, :-1]
        outcome_nll = -outcome_preds.gather(
            2,
            outcome_targets.unsqueeze(-1),
        ).squeeze(-1)

        joint = tool_nll + self.outcome_weight * outcome_nll

        nll_full = torch.zeros_like(tool_ids, dtype=torch.float)
        nll_full[:, 1:] = joint
        nll_full = nll_full * mask.float()
        return nll_full

    def training_step(
        self,
        tool_ids: torch.Tensor,
        args_hashes: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
        outcome_hashes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute training loss (next-token prediction, cross-entropy).

        When use_outcome=True and outcome_hashes is provided, the loss
        is: tool_ce + outcome_weight * outcome_ce (joint prediction).

        Returns:
            Scalar loss.
        """
        if self.use_outcome and outcome_hashes is not None:
            return self._training_step_joint(
                tool_ids,
                args_hashes,
                mask,
                outcome_hashes,
            )

        logits = self.forward(tool_ids, args_hashes, mask)

        shift_logits = logits[:, :-1].contiguous()
        shift_targets = tool_ids[:, 1:].contiguous()
        shift_mask = mask[:, 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, self.vocab_size),
            shift_targets.view(-1),
            reduction="none",
        )
        loss = loss.view(shift_targets.shape)
        loss = (loss * shift_mask.float()).sum() / shift_mask.float().sum().clamp(min=1)
        return loss

    def _training_step_joint(
        self,
        tool_ids: torch.Tensor,
        args_hashes: torch.Tensor,
        mask: torch.Tensor,
        outcome_hashes: torch.Tensor,
    ) -> torch.Tensor:
        """Joint training: predict next tool + next outcome."""
        tool_logits, outcome_logits = self.forward_joint(
            tool_ids,
            args_hashes,
            mask,
            outcome_hashes,
        )

        shift_mask = mask[:, 1:].contiguous()
        denom = shift_mask.float().sum().clamp(min=1)

        tool_loss = F.cross_entropy(
            tool_logits[:, :-1].contiguous().view(-1, self.vocab_size),
            tool_ids[:, 1:].contiguous().view(-1),
            reduction="none",
        ).view(tool_ids[:, 1:].shape)
        tool_loss = (tool_loss * shift_mask.float()).sum() / denom

        n_outcome_buckets = outcome_logits.size(-1)
        outcome_loss = F.cross_entropy(
            outcome_logits[:, :-1].contiguous().view(-1, n_outcome_buckets),
            outcome_hashes[:, 1:].contiguous().view(-1),
            reduction="none",
        ).view(outcome_hashes[:, 1:].shape)
        outcome_loss = (outcome_loss * shift_mask.float()).sum() / denom

        return tool_loss + self.outcome_weight * outcome_loss

    def trace_score(
        self,
        tool_ids: torch.Tensor,
        args_hashes: torch.Tensor,
        mask: torch.Tensor,
        outcome_hashes: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute per-trace anomaly score (mean NLL over real tokens).

        When use_outcome=True and outcome_hashes is provided, uses
        the joint NLL (tool + outcome prediction).

        Returns:
            scores: (B,) scalar anomaly scores per trace.
        """
        if self.use_outcome and outcome_hashes is not None:
            nll = self.compute_joint_nll(
                tool_ids,
                args_hashes,
                mask,
                outcome_hashes,
            )
        else:
            nll = self.compute_nll(tool_ids, args_hashes, mask, outcome_hashes)
        lengths = mask.float().sum(dim=1).clamp(min=1)
        return nll.sum(dim=1) / lengths


def build_encoder(
    *,
    vocab_size: int = 22,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 3,
    d_ff: int = 512,
    dropout: float = 0.1,
    use_args: bool = True,
) -> TransformerTraceEncoder:
    """Construct the trace encoder model."""
    return TransformerTraceEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        use_args=use_args,
    )


def build_large_encoder(
    *,
    vocab_size: int = 22,
    dropout: float = 0.1,
    use_args: bool = False,
    variant: str = "wide",
) -> TransformerTraceEncoder:
    """Construct an iso-parameter large sequence-only encoder (~11M params).

    Used as a capacity-matched control to isolate the effect of
    consequence-aware scoring from raw parameter count. Both variants
    have ~11.1M params, matching the consequence-aware encoder's total.

    Args:
        variant: "wide" = d_model=480, n_layers=4, d_ff=1920 (~11.1M)
                 "deep" = d_model=320, n_layers=9, d_ff=1280 (~11.1M)
    """
    if variant == "wide":
        d_model, n_heads, n_layers, d_ff = 480, 8, 4, 1920
    elif variant == "deep":
        d_model, n_heads, n_layers, d_ff = 320, 8, 9, 1280
    else:
        raise ValueError(f"Unknown variant: {variant!r}. Use 'wide' or 'deep'.")

    return TransformerTraceEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        use_args=use_args,
    )


def build_consequence_encoder(
    *,
    vocab_size: int = 22,
    d_model: int = 128,
    n_heads: int = 4,
    n_layers: int = 3,
    d_ff: int = 512,
    dropout: float = 0.1,
    use_args: bool = False,
    outcome_weight: float = 0.5,
) -> TransformerTraceEncoder:
    """Construct a consequence-aware trace encoder.

    The consequence-aware encoder predicts both the next tool and the
    next outcome. The joint anomaly score combines tool prediction NLL
    with outcome prediction NLL, weighted by outcome_weight.

    Args:
        outcome_weight: Weight for outcome NLL in the joint score.
            0.0 = tool-only, 1.0 = equal weight.
    """
    return TransformerTraceEncoder(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        dropout=dropout,
        use_args=use_args,
        use_outcome=True,
        outcome_weight=outcome_weight,
    )

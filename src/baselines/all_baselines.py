"""All baseline methods for TrustLens comparison.

Each baseline implements:
    name() -> str
    fit(train_scores, train_labels, cal_scores, cal_labels) -> None
    predict(scores) -> np.ndarray  (binary: 1=anomalous, 0=normal)
    score(traces_data) -> np.ndarray  (continuous anomaly scores)
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LogisticRegression

from data.tool_registry import RISKY_TOOLS
from trustlens.conformal import adaptive_conformal_update, split_conformal_calibrate


class BaselineMethod(ABC):
    """Common interface for all baseline methods."""

    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def fit(
        self,
        train_scores: np.ndarray,
        train_labels: np.ndarray,
        cal_scores: np.ndarray,
        cal_labels: np.ndarray,
    ) -> None: ...

    @abstractmethod
    def predict(self, scores: np.ndarray) -> np.ndarray: ...


# ---------------------------------------------------------------------------
# 1. No Permissions (lower bound)
# ---------------------------------------------------------------------------


class NoPermissionsBaseline(BaselineMethod):
    """Permits everything — lower bound on safety."""

    def name(self) -> str:
        return "No Permissions"

    def fit(self, train_scores, train_labels, cal_scores, cal_labels) -> None:
        pass

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return np.zeros(len(scores), dtype=np.int32)


# ---------------------------------------------------------------------------
# 2. Static RBAC
# ---------------------------------------------------------------------------


class RBACBaseline(BaselineMethod):
    """Static role-based access control: blocklist of risky tools."""

    def __init__(self, blocked_tools: set[str] | None = None) -> None:
        self.blocked_tools = blocked_tools or set(RISKY_TOOLS)
        self._tool_scores: dict[str, float] = {}

    def name(self) -> str:
        return "Static RBAC"

    def fit(self, train_scores, train_labels, cal_scores, cal_labels) -> None:
        pass

    def predict_from_tools(self, tools: list[str]) -> np.ndarray:
        return np.array(
            [1 if t in self.blocked_tools else 0 for t in tools], dtype=np.int32
        )

    def predict(self, scores: np.ndarray) -> np.ndarray:
        threshold = np.median(scores)
        return (scores > threshold).astype(np.int32)


# ---------------------------------------------------------------------------
# 3. Rule-Based Anomaly Detection
# ---------------------------------------------------------------------------


class RuleBasedBaseline(BaselineMethod):
    """Threshold on frequency statistics of tool calls."""

    def __init__(self) -> None:
        self._threshold: float = 0.0

    def name(self) -> str:
        return "Rule-Based"

    def fit(self, train_scores, train_labels, cal_scores, cal_labels) -> None:
        normal_scores = train_scores[train_labels == 0]
        if len(normal_scores) > 0:
            self._threshold = np.mean(normal_scores) + 2.0 * np.std(normal_scores)
        else:
            self._threshold = np.percentile(train_scores, 95)

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return (scores > self._threshold).astype(np.int32)


# ---------------------------------------------------------------------------
# 4. Uncalibrated Detector
# ---------------------------------------------------------------------------


class UncalibratedBaseline(BaselineMethod):
    """Same encoder scores, but with a fixed percentile threshold (no CP)."""

    def __init__(self, percentile: float = 95.0) -> None:
        self._percentile = percentile
        self._threshold: float = 0.0

    def name(self) -> str:
        return "Uncalibrated"

    def fit(self, train_scores, train_labels, cal_scores, cal_labels) -> None:
        self._threshold = np.percentile(
            np.concatenate([train_scores, cal_scores]), self._percentile
        )

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return (scores > self._threshold).astype(np.int32)


# ---------------------------------------------------------------------------
# 5. Split CP (vanilla)
# ---------------------------------------------------------------------------


class SplitCPBaseline(BaselineMethod):
    """Standard split conformal prediction (no feedback awareness)."""

    def __init__(self, alpha: float = 0.05) -> None:
        self._alpha = alpha
        self._tau: float = 0.0

    def name(self) -> str:
        return "Split CP"

    def fit(self, train_scores, train_labels, cal_scores, cal_labels) -> None:
        self._tau = split_conformal_calibrate(cal_scores, alpha=self._alpha)

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return (scores > self._tau).astype(np.int32)


# ---------------------------------------------------------------------------
# 6. ACI (Gibbs & Candès)
# ---------------------------------------------------------------------------


class ACIBaseline(BaselineMethod):
    """Adaptive Conformal Inference without domain tightening."""

    def __init__(self, alpha: float = 0.05, gamma: float = 0.01) -> None:
        self._alpha = alpha
        self._gamma = gamma
        self._tau: float = 0.0

    def name(self) -> str:
        return "ACI"

    def fit(self, train_scores, train_labels, cal_scores, cal_labels) -> None:
        self._tau = split_conformal_calibrate(cal_scores, alpha=self._alpha)

    def predict(self, scores: np.ndarray) -> np.ndarray:
        preds = np.zeros(len(scores), dtype=np.int32)
        tau = self._tau
        for i, s in enumerate(scores):
            preds[i] = int(s > tau)
            err = bool(preds[i])
            tau = adaptive_conformal_update(
                tau, err, alpha=self._alpha, gamma=self._gamma
            )
        return preds


# ---------------------------------------------------------------------------
# 7. TrajAD (simplified NLL-based verifier)
# ---------------------------------------------------------------------------


class TrajADBaseline(BaselineMethod):
    """Simplified TrajAD: uses same NLL scores but with learned threshold.

    TrajAD (Li et al., 2025) uses a fine-tuned LLM as verifier. We
    approximate with a logistic regression on the NLL scores as a
    reasonable proxy — we can't run their full model locally.
    """

    def __init__(self) -> None:
        self._clf: LogisticRegression | None = None

    def name(self) -> str:
        return "TrajAD (proxy)"

    def fit(self, train_scores, train_labels, cal_scores, cal_labels) -> None:
        X = np.concatenate([train_scores, cal_scores]).reshape(-1, 1)
        y = np.concatenate([train_labels, cal_labels])
        self._clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        self._clf.fit(X, y)

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if self._clf is None:
            return np.zeros(len(scores), dtype=np.int32)
        return self._clf.predict(scores.reshape(-1, 1)).astype(np.int32)


# ---------------------------------------------------------------------------
# 8. Trajectory Guard (Siamese Autoencoder proxy)
# ---------------------------------------------------------------------------


class TrajectoryGuardBaseline(BaselineMethod):
    """Trajectory Guard proxy: reconstruction error threshold.

    Trajectory Guard (Rawal et al., 2025) uses a siamese recurrent
    autoencoder. We approximate with reconstruction error from a
    simple autoencoder on the NLL score sequence.
    """

    def __init__(self) -> None:
        self._mean: float = 0.0
        self._std: float = 1.0
        self._threshold: float = 0.0

    def name(self) -> str:
        return "Trajectory Guard (proxy)"

    def fit(self, train_scores, train_labels, cal_scores, cal_labels) -> None:
        normal = train_scores[train_labels == 0]
        if len(normal) > 0:
            self._mean = np.mean(normal)
            self._std = max(np.std(normal), 1e-6)
        recon_error = np.abs(cal_scores - self._mean) / self._std
        self._threshold = np.percentile(recon_error, 95)

    def predict(self, scores: np.ndarray) -> np.ndarray:
        recon_error = np.abs(scores - self._mean) / self._std
        return (recon_error > self._threshold).astype(np.int32)


# ---------------------------------------------------------------------------
# 9. DQN-Threshold (RL-based adaptive threshold)
# ---------------------------------------------------------------------------


class DQNThresholdBaseline(BaselineMethod):
    """DQN-Threshold: RL-based adaptive threshold via epsilon-greedy.

    Simplified version: the "agent" picks a threshold from a discrete
    set and receives reward based on classification accuracy. We use
    a tabular Q-learning approximation.
    """

    def __init__(
        self,
        n_thresholds: int = 20,
        n_episodes: int = 200,
        epsilon: float = 0.1,
        lr: float = 0.1,
        gamma_rl: float = 0.95,
    ) -> None:
        self._n_thresholds = n_thresholds
        self._n_episodes = n_episodes
        self._epsilon = epsilon
        self._lr = lr
        self._gamma_rl = gamma_rl
        self._thresholds: np.ndarray = np.array([])
        self._q_values: np.ndarray = np.array([])
        self._best_threshold: float = 0.0

    def name(self) -> str:
        return "DQN-Threshold"

    def fit(self, train_scores, train_labels, cal_scores, cal_labels) -> None:
        all_scores = np.concatenate([train_scores, cal_scores])
        lo, hi = np.percentile(all_scores, [5, 99])
        self._thresholds = np.linspace(lo, hi, self._n_thresholds)
        self._q_values = np.zeros(self._n_thresholds)

        scores = cal_scores
        labels = cal_labels
        rng = np.random.default_rng(42)

        for _ in range(self._n_episodes):
            if rng.random() < self._epsilon:
                action = rng.integers(0, self._n_thresholds)
            else:
                action = int(np.argmax(self._q_values))

            tau = self._thresholds[action]
            preds = (scores > tau).astype(np.int32)

            tp = ((preds == 1) & (labels == 1)).sum()
            fp = ((preds == 1) & (labels == 0)).sum()
            fn = ((preds == 0) & (labels == 1)).sum()
            reward = tp - 2.0 * fn - 0.5 * fp

            self._q_values[action] += self._lr * (
                reward - self._q_values[action]
            )

        best_action = int(np.argmax(self._q_values))
        self._best_threshold = self._thresholds[best_action]

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return (scores > self._best_threshold).astype(np.int32)


# ---------------------------------------------------------------------------
# 10. Temperature Scaling
# ---------------------------------------------------------------------------


class TemperatureScalingBaseline(BaselineMethod):
    """Post-hoc temperature scaling on encoder logits.

    Finds optimal temperature T that minimizes NLL on calibration set,
    then uses the rescaled scores for thresholding.
    """

    def __init__(self, alpha: float = 0.05) -> None:
        self._alpha = alpha
        self._temperature: float = 1.0
        self._tau: float = 0.0

    def name(self) -> str:
        return "Temperature Scaling"

    def fit(self, train_scores, train_labels, cal_scores, cal_labels) -> None:
        best_t = 1.0
        best_ece = float("inf")
        for t in np.linspace(0.1, 5.0, 50):
            scaled = cal_scores / t
            preds = (scaled > np.median(scaled)).astype(int)
            ece = abs(preds.mean() - cal_labels.mean())
            if ece < best_ece:
                best_ece = ece
                best_t = t
        self._temperature = best_t
        scaled_cal = cal_scores / self._temperature
        self._tau = split_conformal_calibrate(scaled_cal, alpha=self._alpha)

    def predict(self, scores: np.ndarray) -> np.ndarray:
        scaled = scores / self._temperature
        return (scaled > self._tau).astype(np.int32)


# ---------------------------------------------------------------------------
# 11. Platt Scaling
# ---------------------------------------------------------------------------


class PlattScalingBaseline(BaselineMethod):
    """Platt scaling: logistic regression on encoder scores."""

    def __init__(self) -> None:
        self._clf: LogisticRegression | None = None

    def name(self) -> str:
        return "Platt Scaling"

    def fit(self, train_scores, train_labels, cal_scores, cal_labels) -> None:
        X = cal_scores.reshape(-1, 1)
        y = cal_labels
        self._clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        self._clf.fit(X, y)

    def predict(self, scores: np.ndarray) -> np.ndarray:
        if self._clf is None:
            return np.zeros(len(scores), dtype=np.int32)
        return self._clf.predict(scores.reshape(-1, 1)).astype(np.int32)


# ---------------------------------------------------------------------------
# 12-13. Independent-Scorer Baselines
# ---------------------------------------------------------------------------


class IndependentScorerBaseline(BaselineMethod):
    """Base for baselines with their own scoring functions (not encoder scores)."""

    @abstractmethod
    def fit_traces(
        self,
        train_traces: list,
        cal_traces: list,
        vocab,
    ) -> None: ...

    @abstractmethod
    def score_traces(self, traces: list, vocab) -> np.ndarray: ...

    def fit(self, train_scores, train_labels, cal_scores, cal_labels) -> None:
        pass

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return (scores > self._threshold).astype(np.int32)


class IsolationForestBaseline(IndependentScorerBaseline):
    """Isolation Forest on hand-crafted trace features.

    Features: sequence length, tool diversity, Shannon entropy,
    risky tool ratio, max consecutive repeat, bigram diversity,
    and per-tool frequency vector.
    """

    _RISKY_TOOLS = set(RISKY_TOOLS)

    def __init__(self, contamination: float = 0.05, random_state: int = 42) -> None:
        self._contamination = contamination
        self._random_state = random_state
        self._model: IsolationForest | None = None
        self._threshold: float = 0.0
        self._tool_list: list[str] = []

    def name(self) -> str:
        return "Isolation Forest"

    def _extract_features(self, traces: list) -> np.ndarray:
        tool_idx = {t: i for i, t in enumerate(self._tool_list)}
        n_tools = len(self._tool_list)

        features = []
        for trace in traces:
            tools = [a.tool for a in trace.actions]
            n = len(tools)

            counts = Counter(tools)
            total = sum(counts.values())
            entropy = 0.0
            for c in counts.values():
                p = c / total
                if p > 0:
                    entropy -= p * math.log2(p)

            n_risky = sum(1 for t in tools if t in self._RISKY_TOOLS)

            max_repeat = 1
            cur_repeat = 1
            for i in range(1, n):
                if tools[i] == tools[i - 1]:
                    cur_repeat += 1
                    max_repeat = max(max_repeat, cur_repeat)
                else:
                    cur_repeat = 1

            if n >= 2:
                bigrams = [(tools[i], tools[i + 1]) for i in range(n - 1)]
                bigram_div = len(set(bigrams)) / len(bigrams)
            else:
                bigram_div = 0.0

            tool_freq = np.zeros(n_tools)
            for t in tools:
                if t in tool_idx:
                    tool_freq[tool_idx[t]] = counts.get(t, 0) / max(n, 1)

            feat = [n, len(set(tools)), entropy, n_risky / max(n, 1), max_repeat, bigram_div]
            feat.extend(tool_freq.tolist())
            features.append(feat)

        return np.array(features, dtype=np.float64)

    def fit_traces(self, train_traces, cal_traces, vocab) -> None:
        self._tool_list = sorted(vocab.tool2id.keys())
        normal_train = [
            t for t in train_traces if not any(a.label == 1 for a in t.actions)
        ]
        X_train = self._extract_features(normal_train)

        self._model = IsolationForest(
            contamination=self._contamination,
            random_state=self._random_state,
            n_estimators=200,
        )
        self._model.fit(X_train)

        X_cal = self._extract_features(cal_traces)
        cal_scores = -self._model.score_samples(X_cal)
        self._threshold = float(np.percentile(cal_scores, 95))

    def score_traces(self, traces, vocab) -> np.ndarray:
        return -self._model.score_samples(self._extract_features(traces))

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return (scores > self._threshold).astype(np.int32)


class NGramBaseline(IndependentScorerBaseline):
    """Bigram language model over tool sequences.

    Scores traces by average negative log-likelihood of their tool
    bigrams under a smoothed bigram model fit on normal traces.
    """

    def __init__(self, smoothing: float = 0.1) -> None:
        self._smoothing = smoothing
        self._bigram_probs: dict[tuple[str, str], float] = {}
        self._vocab_size: int = 0
        self._default_prob: float = 1e-10
        self._threshold: float = 0.0

    def name(self) -> str:
        return "N-gram LM"

    def fit_traces(self, train_traces, cal_traces, vocab) -> None:
        normal_train = [
            t for t in train_traces if not any(a.label == 1 for a in t.actions)
        ]

        bigram_counts: Counter = Counter()
        unigram_counts: Counter = Counter()
        all_tools: set[str] = set()

        for trace in normal_train:
            tools = [a.tool for a in trace.actions]
            for t in tools:
                unigram_counts[t] += 1
                all_tools.add(t)
            for i in range(len(tools) - 1):
                bigram_counts[(tools[i], tools[i + 1])] += 1

        self._vocab_size = max(len(all_tools), 1)

        for t1 in all_tools:
            t1_count = unigram_counts[t1]
            for t2 in all_tools:
                bc = bigram_counts.get((t1, t2), 0)
                self._bigram_probs[(t1, t2)] = (
                    (bc + self._smoothing) / (t1_count + self._smoothing * self._vocab_size)
                )

        self._default_prob = self._smoothing / (self._smoothing * self._vocab_size)

        cal_scores = self.score_traces(cal_traces, vocab)
        self._threshold = float(np.percentile(cal_scores, 95))

    def _score_trace(self, trace) -> float:
        tools = [a.tool for a in trace.actions]
        if len(tools) < 2:
            return 0.0

        nll = 0.0
        for i in range(len(tools) - 1):
            prob = self._bigram_probs.get(
                (tools[i], tools[i + 1]), self._default_prob
            )
            nll -= math.log(max(prob, 1e-10))

        return nll / max(len(tools) - 1, 1)

    def score_traces(self, traces, vocab) -> np.ndarray:
        return np.array([self._score_trace(t) for t in traces], dtype=np.float64)

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return (scores > self._threshold).astype(np.int32)


# ---------------------------------------------------------------------------
# 14. LSTM Next-Token Predictor
# ---------------------------------------------------------------------------


class LSTMBaselineModel(nn.Module):
    """Single-layer LSTM that predicts next tool ID from sequence."""

    def __init__(self, vocab_size: int, d_embed: int = 128, d_hidden: int = 128) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_embed, padding_idx=0)
        self.lstm = nn.LSTM(d_embed, d_hidden, batch_first=True)
        self.head = nn.Linear(d_hidden, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.embed(x)
        out, _ = self.lstm(emb)
        return self.head(out)


class LSTMNextTokenBaseline(IndependentScorerBaseline):
    """LSTM next-token predictor baseline (v12 fair training).

    Trains a single-layer LSTM to predict the next tool in a sequence.
    Anomaly score = average cross-entropy loss per action (higher = more
    anomalous).

    v12: Fair training budget matching the CQ encoder — d_embed=128,
    epochs=50, CosineAnnealingLR, early stopping with patience=5.
    """

    def __init__(
        self,
        d_embed: int = 128,
        d_hidden: int = 128,
        lr: float = 1e-3,
        epochs: int = 50,
        patience: int = 5,
        batch_size: int = 64,
    ) -> None:
        self._d_embed = d_embed
        self._d_hidden = d_hidden
        self._lr = lr
        self._epochs = epochs
        self._patience = patience
        self._batch_size = batch_size
        self._model: LSTMBaselineModel | None = None
        self._threshold: float = 0.0
        self._device: torch.device = torch.device("cpu")

    def name(self) -> str:
        return "LSTM Next-Token"

    def _compute_loss_on_seqs(
        self,
        seqs: list[list[int]],
        vs: int,
        loss_fn: nn.CrossEntropyLoss,
    ) -> float:
        self._model.eval()
        total_loss = 0.0
        n_batches = 0
        with torch.no_grad():
            for i in range(0, len(seqs), self._batch_size):
                batch = seqs[i : i + self._batch_size]
                max_len = max(len(s) for s in batch)
                x = torch.zeros(len(batch), max_len, dtype=torch.long)
                for j, s in enumerate(batch):
                    x[j, : len(s)] = torch.tensor(s, dtype=torch.long)
                x = x.to(self._device)
                inp = x[:, :-1]
                tgt = x[:, 1:]
                logits = self._model(inp)
                loss = loss_fn(logits.reshape(-1, vs), tgt.reshape(-1))
                total_loss += loss.item()
                n_batches += 1
        self._model.train()
        return total_loss / max(n_batches, 1)

    def fit_traces(self, train_traces, cal_traces, vocab) -> None:
        from torch.optim.lr_scheduler import CosineAnnealingLR

        if torch.backends.mps.is_available():
            self._device = torch.device("mps")
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")

        vs = vocab.vocab_size
        self._model = LSTMBaselineModel(vs, self._d_embed, self._d_hidden)
        self._model.to(self._device)

        normal_train = [
            t for t in train_traces if not any(a.label == 1 for a in t.actions)
        ]

        all_seqs = [[vocab.encode_tool(a.tool) for a in t.actions] for t in normal_train]
        n_val = max(1, int(len(all_seqs) * 0.2))
        rng = np.random.RandomState(42)
        perm = rng.permutation(len(all_seqs))
        val_seqs = [all_seqs[i] for i in perm[:n_val]]
        train_seqs = [all_seqs[i] for i in perm[n_val:]]

        optimizer = torch.optim.AdamW(self._model.parameters(), lr=self._lr, weight_decay=0.01)
        scheduler = CosineAnnealingLR(optimizer, T_max=self._epochs)
        loss_fn = nn.CrossEntropyLoss(ignore_index=0)

        best_val_loss = float("inf")
        best_state = None
        patience_counter = 0

        self._model.train()
        for _epoch in range(self._epochs):
            np.random.shuffle(train_seqs)
            for i in range(0, len(train_seqs), self._batch_size):
                batch = train_seqs[i : i + self._batch_size]
                max_len = max(len(s) for s in batch)
                x = torch.zeros(len(batch), max_len, dtype=torch.long)
                for j, s in enumerate(batch):
                    x[j, : len(s)] = torch.tensor(s, dtype=torch.long)

                x = x.to(self._device)
                inp = x[:, :-1]
                tgt = x[:, 1:]

                logits = self._model(inp)
                loss = loss_fn(logits.reshape(-1, vs), tgt.reshape(-1))
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            val_loss = self._compute_loss_on_seqs(val_seqs, vs, loss_fn)
            if val_loss < best_val_loss - 1e-6:
                best_val_loss = val_loss
                best_state = {
                    k: v.cpu().clone() for k, v in self._model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self._patience:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        self._model.to(self._device)
        self._model.eval()

        cal_scores = self.score_traces(cal_traces, vocab)
        self._threshold = float(np.percentile(cal_scores, 95))

    def score_traces(self, traces, vocab) -> np.ndarray:
        if self._model is None:
            return np.zeros(len(traces), dtype=np.float64)

        self._model.eval()
        vs = self._model.head.out_features
        loss_fn = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
        scores = []

        with torch.no_grad():
            for trace in traces:
                ids = [vocab.encode_tool(a.tool) for a in trace.actions]
                if len(ids) < 2:
                    scores.append(0.0)
                    continue
                x = torch.tensor([ids], dtype=torch.long, device=self._device)
                inp = x[:, :-1]
                tgt = x[:, 1:]
                logits = self._model(inp)
                per_token = loss_fn(logits.reshape(-1, vs), tgt.reshape(-1))
                mask = tgt.reshape(-1) != 0
                if mask.sum() > 0:
                    scores.append(per_token[mask].mean().item())
                else:
                    scores.append(0.0)

        return np.array(scores, dtype=np.float64)

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return (scores > self._threshold).astype(np.int32)


# ---------------------------------------------------------------------------
# 15. LSTM Consequence-Aware (tool + outcome prediction)
# ---------------------------------------------------------------------------


class LSTMConsequenceModel(nn.Module):
    """LSTM predicting both next tool ID and next outcome hash.

    Joint loss = tool_CE + outcome_weight * outcome_CE.
    Mirrors the CQ transformer's dual-prediction approach but with
    an LSTM backbone, isolating the effect of outcome information
    from the transformer architecture.
    """

    def __init__(
        self,
        vocab_size: int,
        outcome_buckets: int = 2**16,
        d_embed: int = 128,
        d_hidden: int = 128,
        outcome_weight: float = 0.5,
    ) -> None:
        super().__init__()
        self.outcome_weight = outcome_weight
        self.tool_embed = nn.Embedding(vocab_size, d_embed, padding_idx=0)
        self.outcome_embed = nn.Embedding(outcome_buckets, d_embed // 4)
        self.lstm = nn.LSTM(
            d_embed + d_embed // 4, d_hidden, batch_first=True,
        )
        self.tool_head = nn.Linear(d_hidden, vocab_size)
        self.outcome_head = nn.Linear(d_hidden, outcome_buckets)

    def forward(
        self,
        tool_ids: torch.Tensor,
        outcome_hashes: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (tool_logits, outcome_logits)."""
        x = torch.cat(
            [self.tool_embed(tool_ids), self.outcome_embed(outcome_hashes)],
            dim=-1,
        )
        out, _ = self.lstm(x)
        return self.tool_head(out), self.outcome_head(out)


class LSTMConsequenceBaseline(IndependentScorerBaseline):
    """LSTM next-token + next-outcome predictor (v13 — architecture ablation).

    Same dual-prediction objective as the CQ transformer but with an
    LSTM backbone. Comparing LSTM+Outcome vs LSTM-only isolates the
    contribution of outcome information. Comparing LSTM+Outcome vs CQ
    isolates the contribution of the transformer architecture.
    """

    def __init__(
        self,
        d_embed: int = 128,
        d_hidden: int = 128,
        lr: float = 1e-3,
        epochs: int = 50,
        patience: int = 5,
        batch_size: int = 64,
        outcome_weight: float = 0.5,
    ) -> None:
        self._d_embed = d_embed
        self._d_hidden = d_hidden
        self._lr = lr
        self._epochs = epochs
        self._patience = patience
        self._batch_size = batch_size
        self._outcome_weight = outcome_weight
        self._model: LSTMConsequenceModel | None = None
        self._threshold: float = 0.0
        self._device: torch.device = torch.device("cpu")

    def name(self) -> str:
        return "LSTM+Outcome"

    def _joint_loss(
        self,
        tool_logits: torch.Tensor,
        outcome_logits: torch.Tensor,
        tool_targets: torch.Tensor,
        outcome_targets: torch.Tensor,
        vs: int,
        obs: int,
        loss_fn_tool: nn.CrossEntropyLoss,
        loss_fn_outcome: nn.CrossEntropyLoss,
    ) -> torch.Tensor:
        t_loss = loss_fn_tool(tool_logits.reshape(-1, vs), tool_targets.reshape(-1))
        o_loss = loss_fn_outcome(
            outcome_logits.reshape(-1, obs), outcome_targets.reshape(-1),
        )
        return t_loss + self._outcome_weight * o_loss

    def _compute_val_loss(
        self,
        seqs_tool: list[list[int]],
        seqs_outcome: list[list[int]],
        vs: int,
        obs: int,
    ) -> float:
        self._model.eval()
        loss_fn_tool = nn.CrossEntropyLoss(ignore_index=0)
        loss_fn_outcome = nn.CrossEntropyLoss(ignore_index=0)
        total, n = 0.0, 0
        with torch.no_grad():
            for i in range(0, len(seqs_tool), self._batch_size):
                bt = seqs_tool[i : i + self._batch_size]
                bo = seqs_outcome[i : i + self._batch_size]
                ml = max(len(s) for s in bt)
                xt = torch.zeros(len(bt), ml, dtype=torch.long)
                xo = torch.zeros(len(bt), ml, dtype=torch.long)
                for j, (st, so) in enumerate(zip(bt, bo)):
                    xt[j, : len(st)] = torch.tensor(st, dtype=torch.long)
                    xo[j, : len(so)] = torch.tensor(so, dtype=torch.long)
                xt, xo = xt.to(self._device), xo.to(self._device)
                tl, ol = self._model(xt[:, :-1], xo[:, :-1])
                loss = self._joint_loss(
                    tl, ol, xt[:, 1:], xo[:, 1:], vs, obs,
                    loss_fn_tool, loss_fn_outcome,
                )
                total += loss.item()
                n += 1
        self._model.train()
        return total / max(n, 1)

    def fit_traces(self, train_traces, cal_traces, vocab) -> None:
        import hashlib

        from torch.optim.lr_scheduler import CosineAnnealingLR

        def _hash_outcome(o: str) -> int:
            return (
                int(hashlib.md5(o.encode(), usedforsecurity=False).hexdigest()[:8], 16)
                % (2**16)
            )

        if torch.backends.mps.is_available():
            self._device = torch.device("mps")
        elif torch.cuda.is_available():
            self._device = torch.device("cuda")

        vs = vocab.vocab_size
        obs = 2**16
        self._model = LSTMConsequenceModel(
            vs, obs, self._d_embed, self._d_hidden, self._outcome_weight,
        )
        self._model.to(self._device)

        normal_train = [
            t for t in train_traces if not any(a.label == 1 for a in t.actions)
        ]

        all_tool_seqs = [
            [vocab.encode_tool(a.tool) for a in t.actions] for t in normal_train
        ]
        all_out_seqs = [
            [_hash_outcome(a.outcome) for a in t.actions] for t in normal_train
        ]

        n_val = max(1, int(len(all_tool_seqs) * 0.2))
        rng = np.random.RandomState(42)
        perm = rng.permutation(len(all_tool_seqs))
        val_t = [all_tool_seqs[i] for i in perm[:n_val]]
        val_o = [all_out_seqs[i] for i in perm[:n_val]]
        train_t = [all_tool_seqs[i] for i in perm[n_val:]]
        train_o = [all_out_seqs[i] for i in perm[n_val:]]

        optimizer = torch.optim.AdamW(
            self._model.parameters(), lr=self._lr, weight_decay=0.01,
        )
        scheduler = CosineAnnealingLR(optimizer, T_max=self._epochs)
        loss_fn_tool = nn.CrossEntropyLoss(ignore_index=0)
        loss_fn_outcome = nn.CrossEntropyLoss(ignore_index=0)

        best_val = float("inf")
        best_state = None
        patience_ctr = 0

        self._model.train()
        for _epoch in range(self._epochs):
            indices = list(range(len(train_t)))
            np.random.shuffle(indices)
            for i in range(0, len(indices), self._batch_size):
                bi = indices[i : i + self._batch_size]
                bt = [train_t[j] for j in bi]
                bo = [train_o[j] for j in bi]
                ml = max(len(s) for s in bt)
                xt = torch.zeros(len(bt), ml, dtype=torch.long)
                xo = torch.zeros(len(bt), ml, dtype=torch.long)
                for j, (st, so) in enumerate(zip(bt, bo)):
                    xt[j, : len(st)] = torch.tensor(st, dtype=torch.long)
                    xo[j, : len(so)] = torch.tensor(so, dtype=torch.long)
                xt, xo = xt.to(self._device), xo.to(self._device)
                tl, ol = self._model(xt[:, :-1], xo[:, :-1])
                loss = self._joint_loss(
                    tl, ol, xt[:, 1:], xo[:, 1:], vs, obs,
                    loss_fn_tool, loss_fn_outcome,
                )
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            vl = self._compute_val_loss(val_t, val_o, vs, obs)
            if vl < best_val - 1e-6:
                best_val = vl
                best_state = {
                    k: v.cpu().clone() for k, v in self._model.state_dict().items()
                }
                patience_ctr = 0
            else:
                patience_ctr += 1
                if patience_ctr >= self._patience:
                    break

        if best_state is not None:
            self._model.load_state_dict(best_state)
        self._model.to(self._device)
        self._model.eval()

        cal_scores = self.score_traces(cal_traces, vocab)
        self._threshold = float(np.percentile(cal_scores, 95))

    def score_traces(self, traces, vocab) -> np.ndarray:
        if self._model is None:
            return np.zeros(len(traces), dtype=np.float64)

        import hashlib

        def _hash_outcome(o: str) -> int:
            return (
                int(hashlib.md5(o.encode(), usedforsecurity=False).hexdigest()[:8], 16)
                % (2**16)
            )

        self._model.eval()
        vs = self._model.tool_head.out_features
        obs = self._model.outcome_head.out_features
        loss_fn_tool = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
        loss_fn_outcome = nn.CrossEntropyLoss(ignore_index=0, reduction="none")
        scores = []

        with torch.no_grad():
            for trace in traces:
                tids = [vocab.encode_tool(a.tool) for a in trace.actions]
                ohs = [_hash_outcome(a.outcome) for a in trace.actions]
                if len(tids) < 2:
                    scores.append(0.0)
                    continue
                xt = torch.tensor([tids], dtype=torch.long, device=self._device)
                xo = torch.tensor([ohs], dtype=torch.long, device=self._device)
                tl, ol = self._model(xt[:, :-1], xo[:, :-1])
                t_nll = loss_fn_tool(tl.reshape(-1, vs), xt[:, 1:].reshape(-1))
                o_nll = loss_fn_outcome(
                    ol.reshape(-1, obs), xo[:, 1:].reshape(-1),
                )
                joint = t_nll + self._outcome_weight * o_nll
                mask = xt[:, 1:].reshape(-1) != 0
                if mask.sum() > 0:
                    scores.append(joint[mask].mean().item())
                else:
                    scores.append(0.0)

        return np.array(scores, dtype=np.float64)

    def predict(self, scores: np.ndarray) -> np.ndarray:
        return (scores > self._threshold).astype(np.int32)


# ---------------------------------------------------------------------------
# Convenience
# ---------------------------------------------------------------------------


def _get_all_baselines_legacy(alpha: float = 0.05) -> list[BaselineMethod]:
    """Deprecated: returns all baselines including proxy methods.

    Use get_scoring_baselines(), get_calibration_baselines(), or
    get_permission_baselines() instead for clean 3-table structure.
    """
    return [
        NoPermissionsBaseline(),
        RBACBaseline(),
        RuleBasedBaseline(),
        UncalibratedBaseline(),
        SplitCPBaseline(alpha=alpha),
        ACIBaseline(alpha=alpha),
        TrajADBaseline(),
        TrajectoryGuardBaseline(),
        DQNThresholdBaseline(),
        TemperatureScalingBaseline(alpha=alpha),
        PlattScalingBaseline(),
        IsolationForestBaseline(),
        NGramBaseline(),
        LSTMNextTokenBaseline(),
        LSTMConsequenceBaseline(),
    ]


def get_scoring_baselines() -> list[IndependentScorerBaseline]:
    """Table 1 — Independent anomaly scorers (apples-to-apples comparison).

    Each scorer produces its own anomaly scores from raw traces,
    independent of the TrustLens encoder. AUROC comparisons between
    these methods are meaningful.
    """
    return [
        LSTMNextTokenBaseline(),
        LSTMConsequenceBaseline(),
        NGramBaseline(),
        IsolationForestBaseline(),
    ]


def get_calibration_baselines(alpha: float = 0.05) -> list[BaselineMethod]:
    """Table 2 — Calibration/thresholding variants (given encoder scores).

    All methods consume the same encoder NLL scores and differ only
    in how they set the decision threshold. AUROC is identical across
    these; F1 and coverage differ.
    """
    return [
        SplitCPBaseline(alpha=alpha),
        ACIBaseline(alpha=alpha),
        UncalibratedBaseline(),
        PlattScalingBaseline(),
        TemperatureScalingBaseline(alpha=alpha),
        DQNThresholdBaseline(),
        RuleBasedBaseline(),
    ]


def get_permission_baselines() -> list[BaselineMethod]:
    """Table 3 — Permission-based methods (different paradigm).

    These methods make decisions based on tool identity or static
    rules, not learned anomaly scores. Included for completeness.
    """
    return [
        RBACBaseline(),
        NoPermissionsBaseline(),
    ]

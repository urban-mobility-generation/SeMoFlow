from __future__ import annotations

import torch
import torch.nn.functional as F

from semoflow.planner.types import (
    OccurrenceBatch,
    OccurrencePlannerLoss,
    OccurrencePlannerOutput,
)


def _legal_label_smoothed_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    label_smoothing: float,
) -> torch.Tensor:
    log_probabilities = F.log_softmax(logits, dim=-1)
    legal = ~torch.isneginf(logits)
    legal_count = legal.sum(dim=-1)
    if bool((legal_count < 1).any()):
        raise ValueError("every prefix decision must have a legal class")
    target_log_probability = log_probabilities.gather(
        -1, targets[..., None]
    ).squeeze(-1)
    smooth_loss = (
        -log_probabilities.masked_fill(~legal, 0.0).sum(dim=-1) / legal_count
    )
    return (
        -(1.0 - label_smoothing) * target_log_probability
        + label_smoothing * smooth_loss
    )


def occurrence_planner_loss(
    output: OccurrencePlannerOutput,
    batch: OccurrenceBatch,
    *,
    length_weight: float = 1.0,
    gap_weight: float = 1.0,
    prefix_weight: float = 1.0,
    label_smoothing: float = 0.0,
    recurrence_weight: float = 0.0,
) -> OccurrencePlannerLoss:
    if min(length_weight, gap_weight, prefix_weight, recurrence_weight) < 0:
        raise ValueError("loss weights must be non-negative")
    if not 0.0 <= label_smoothing < 1.0:
        raise ValueError("label_smoothing must be in [0, 1)")
    valid = batch.valid_mask
    batch_size = batch.lengths.shape[0]
    length_loss = _legal_label_smoothed_cross_entropy(
        output.length_logits,
        batch.lengths - 1,
        label_smoothing=label_smoothing,
    ).mean()
    gap_loss = (
        _legal_label_smoothed_cross_entropy(
            output.gap_logits[valid],
            batch.target_gap_tokens[valid],
            label_smoothing=label_smoothing,
        ).sum()
        / batch_size
    )
    if output.duration_log_probability is None:
        duration_loss = gap_loss.new_zeros(())
    else:
        if (
            output.duration_log_probability.shape
            != batch.target_gap_tokens.shape
        ):
            raise ValueError(
                "duration log probability must align with gap targets"
            )
        duration_loss = (
            -output.duration_log_probability[valid].sum() / batch_size
        )
    if output.prefix_log_probability is not None:
        if label_smoothing != 0.0:
            raise ValueError(
                "label_smoothing is unsupported with routed prefix likelihood"
            )
        if output.prefix_log_probability.shape != valid.shape:
            raise ValueError(
                "prefix_log_probability must align with valid_mask"
            )
        prefix_loss = -output.prefix_log_probability[valid].sum() / batch_size
    elif output.prefix_logits.ndim == 4:
        prefix_valid = valid[..., None].expand(*output.prefix_logits.shape[:3])
        prefix_decision_losses = _legal_label_smoothed_cross_entropy(
            output.prefix_logits[prefix_valid],
            output.prefix_targets[prefix_valid],
            label_smoothing=label_smoothing,
        )
        prefix_loss = prefix_decision_losses.sum() / batch_size
    else:
        prefix_loss = (
            _legal_label_smoothed_cross_entropy(
                output.prefix_logits[valid],
                output.prefix_targets[valid],
                label_smoothing=label_smoothing,
            ).sum()
            / batch_size
        )
    recurrence_loss = prefix_loss.new_zeros(())
    if recurrence_weight > 0.0:
        if output.retrieve_probability is None:
            raise ValueError(
                "recurrence loss requires an Explore/Retrieve planner"
            )
        if output.retrieve_probability.shape != valid.shape:
            raise ValueError("retrieve_probability must align with valid_mask")
        history = batch.previous_prefix_codes
        targets = batch.target_prefix_codes
        has_history = (history >= 0).all(dim=-1)
        steps = torch.arange(history.shape[1], device=history.device)
        causal_history = steps[None, :, None] <= steps[None, None, :]
        repeats = (
            (history[:, :, None, :] == targets[:, None, :, :]).all(dim=-1)
            & has_history[:, :, None]
            & causal_history
        ).any(dim=1)
        visible_history = (has_history[:, :, None] & causal_history).any(dim=1)
        supervised = valid & visible_history
        route_probability = output.retrieve_probability.clamp(1e-6, 1.0 - 1e-6)
        recurrence_loss = (
            F.binary_cross_entropy(
                route_probability[supervised],
                repeats.to(route_probability.dtype)[supervised],
                reduction="sum",
            )
            / batch_size
        )
    total = (
        length_weight * length_loss
        + gap_weight * (gap_loss + duration_loss)
        + prefix_weight * prefix_loss
        + recurrence_weight * recurrence_loss
    )
    return OccurrencePlannerLoss(
        total=total,
        length=length_loss,
        gap=gap_loss,
        prefix=prefix_loss,
        duration=duration_loss,
        recurrence=recurrence_loss,
    )

from __future__ import annotations

import torch.nn.functional as F

from semoflow.planner.types import (
    OccurrenceBatch,
    OccurrencePlannerLoss,
    OccurrencePlannerOutput,
)


def occurrence_planner_loss(
    output: OccurrencePlannerOutput, batch: OccurrenceBatch
) -> OccurrencePlannerLoss:
    """Per-sequence NLL of the length, gap, and routed prefix choices."""
    valid = batch.valid_mask
    batch_size = batch.lengths.shape[0]
    length_loss = F.cross_entropy(output.length_logits, batch.lengths - 1)
    gap_loss = (
        F.cross_entropy(
            output.gap_logits[valid],
            batch.target_gap_tokens[valid],
            reduction="sum",
        )
        / batch_size
    )
    prefix_loss = -output.prefix_log_probability[valid].sum() / batch_size
    return OccurrencePlannerLoss(
        total=length_loss + gap_loss + prefix_loss,
        length=length_loss,
        gap=gap_loss,
        prefix=prefix_loss,
    )

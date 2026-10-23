import torch
import torch.nn.functional as F


def weighted_bce_with_logits(
    logits: torch.Tensor,
    targets: torch.Tensor,
    pos_w: torch.Tensor,
    neg_w: torch.Tensor,
    reduction: str = "mean",
) -> torch.Tensor:
    """BCE-with-logits with explicit positive/negative coefficients.

    Args:
        logits: Model logits, shape [..., S, C].
        targets: Binary targets, same shape as logits.
        pos_w: Multipliers for positive term (y * log(sigmoid(logits))), broadcastable to logits.
        neg_w: Multipliers for negative term ((1-y) * log(sigmoid(-logits))), broadcastable to logits.
        reduction: One of {"mean", "sum", "none"}.
    """
    pos_term = targets * F.logsigmoid(logits)
    neg_term = (1.0 - targets) * F.logsigmoid(-logits)
    loss = -(pos_w * pos_term + neg_w * neg_term)
    if reduction == "sum":
        return loss.sum()
    if reduction == "none":
        return loss
    if reduction == "mean":
        return loss.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")

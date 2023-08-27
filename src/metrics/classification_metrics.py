import torch


def accuracy(
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        logits: bool = True
):
    if logits:
        predictions = torch.argmax(y_pred, dim=1)
    else:
        predictions = y_pred

    result = torch.sum(predictions == y_true) / len(predictions)

    return result

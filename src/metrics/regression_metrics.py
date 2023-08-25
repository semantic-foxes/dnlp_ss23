import torch


def pearson_correlation(
        y_pred: torch.Tensor,
        y_true: torch.Tensor
):
    pred_mean = torch.mean(y_pred)
    true_mean = torch.mean(y_true)

    numerator = torch.sum((y_pred - pred_mean)*(y_true - true_mean))
    pred_sigma = torch.sum((y_pred - pred_mean)**2)
    true_sigma = torch.sum((y_true - true_mean)**2)

    result = numerator / torch.sqrt(pred_sigma * true_sigma)

    return result


def pearson_correlation_proba(
        y_pred: torch.Tensor,
        y_true: torch.Tensor
):
    y_pred = torch.max(y_pred, dim=1)
    pred_mean = torch.mean(y_pred)
    true_mean = torch.mean(y_true)

    numerator = torch.sum((y_pred - pred_mean)*(y_true - true_mean))
    pred_sigma = torch.sum((y_pred - pred_mean)**2)
    true_sigma = torch.sum((y_true - true_mean)**2)

    result = numerator / torch.sqrt(pred_sigma * true_sigma)

    return result

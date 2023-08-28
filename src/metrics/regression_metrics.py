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
    
    # it could be nan if targets are same: tensor([3.2000, 3.2000])
    return torch.nan_to_num(result)


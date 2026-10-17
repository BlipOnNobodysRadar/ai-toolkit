import torch
import math
import kornia
from typing import Optional
from .laplace_mse_loss import mse_pyramid_loss_2d_non_reduced

def soft_welsch_loss(predictions:torch.Tensor, 
                    targets:torch.Tensor, 
                    reduction: str = "mean", 
                    scale: float = 1.0, 
                    delta: float = 1.0):
    differences = predictions.to(torch.float64) - targets.to(torch.float64)
    loss = torch.arcsinh(4 * (scale * differences**2) / delta) * delta / 4
    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)
    elif reduction == "none":
        loss = loss
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}")
    return loss

def stable_mse_loss(predictions, targets, reduction="mean"):
    differences = predictions.to(torch.float64) - targets.to(torch.float64)
    squared_differences = differences ** 2

    if reduction == "mean":
        loss = torch.mean(squared_differences)
    elif reduction == "sum":
        loss = torch.sum(squared_differences)
    elif reduction == "none":
        loss = squared_differences
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}")
    return loss

def stable_log_cosh_loss(predictions, targets, reduction='mean'):
    diff = predictions.to(torch.float64) - targets.to(torch.float64)
    # For x >= 0
    pos_mask = diff >= 0
    # Compute log(cosh(x)) for positive x
    logcosh_pos = diff + torch.nn.functional.softplus(-2 * diff) - math.log(2)
    # For x < 0
    logcosh_neg = -diff + torch.nn.functional.softplus(2 * diff) - math.log(2)
    # Combine results
    log_cosh = torch.where(pos_mask, logcosh_pos, logcosh_neg)

    if reduction == "mean":
        loss = torch.mean(log_cosh)
    elif reduction == "sum":
        loss = torch.sum(log_cosh)
    elif reduction == "none":
        loss = log_cosh
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}")
    return loss

def stable_msle_loss(predictions, targets, reduction='mean'):
    msle = torch.square(torch.log(targets.to(torch.float64) + 1.0) - torch.log(predictions.to(torch.float64) + 1.0))

    if reduction == "mean":
        loss = torch.mean(msle)
    elif reduction == "sum":
        loss = torch.sum(msle)
    elif reduction == "none":
        loss = msle
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}")
    return loss

def x_sigmoid_loss(predictions, targets, reduction="mean"):
    # Compute at float64
    differences = predictions.to(torch.float64) - targets.to(torch.float64)
    sigmoid_differences = 2 * differences * torch.sigmoid(differences) - differences
    if reduction == "mean":
        loss = torch.mean(sigmoid_differences)
    elif reduction == "sum":
        loss = torch.sum(sigmoid_differences)
    elif reduction == "none":
        loss = sigmoid_differences
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}")
    return loss

def stable_pseudo_huber_loss(predictions, targets, delta=1.0, reduction="mean"):
    """
    Compute the Pseudo-Huber loss between true values and predictions.

    Parameters:
    y_true : array_like
        The ground truth (correct) target values.
    y_pred : array_like
        The predicted target values.
    delta : float, default=1.0
        The parameter delta controls the transition point between the quadratic
        and linear regions of the loss function.

    Returns:
    loss : array_like
        The Pseudo-Huber loss values for each element.
    """
    differences = predictions.to(torch.float64) - targets.to(torch.float64)

    # Compute the loss
    loss = delta**2 * (torch.sqrt(1 + (differences / delta)**2) - 1)
    
    # Apply the specified reduction method
    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)
    elif reduction == "none":
        loss = loss
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}")
    return loss

def scaled_quadratic_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    delta: float = 1.0,
    reduction: str = 'mean'
) -> torch.Tensor:
    r = predictions.to(torch.float64) - targets.to(torch.float64)
    loss = (r / delta)**2
    
    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)
    elif reduction == "none":
        loss = loss
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}")
    return loss

def standard_deviation_loss(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        reduction: str = 'mean',
        eps: float = 1e-100) -> torch.Tensor:
    """
    Calculate standard deviation loss between predicted and true values.
    
    Args:
        predictions (torch.Tensor): Predicted values
        targets (torch.Tensor): True values
        eps (float): Small constant to prevent numerical instability
                    when taking square root
        
    Returns:
        torch.Tensor: The standard deviation loss
    """
    n = predictions.size(0)
    squared_diff = (predictions.to(torch.float64) - targets.to(torch.float64)) ** 2
    mean_squared_diff = torch.sum(squared_diff) / n
    mean_squared_diff_sqrt = torch.sqrt(mean_squared_diff + eps)

    if reduction == "mean":
        loss = torch.mean(mean_squared_diff_sqrt)
    elif reduction == "sum":
        loss = torch.sum(mean_squared_diff_sqrt)
    elif reduction == "none":
        loss = mean_squared_diff_sqrt
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}")
    return loss

def smooth_l2_log_loss(
     predictions: torch.Tensor,
     targets: torch.Tensor,
     delta: float = 1.0,
     reduction: str = 'mean'
 ) -> torch.Tensor:
    """
    Functional version of the smooth l2->log loss.

    Args:
        predictions: Predicted values of shape (*)
        targets: Target values of shape (*), same shape as predictions
        delta: Transition point between L2 and logarithmic behavior
        reduction: Reduction to apply to batch: 'none' | 'mean' | 'sum'
        
    Returns:
        Loss tensor of shape () if reduction is 'mean' or 'sum',
        or same shape as inputs if reduction is 'none'
    """
    r = predictions.to(torch.float64) - targets.to(torch.float64)
    delta_squared = delta ** 2
    loss = 0.5 * delta_squared * torch.log1p(r ** 2 / delta_squared)

    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)
    elif reduction == "none":
        loss = loss
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}")
    return loss

def stable_smooth_l1_loss(predictions, targets, reduction: str = 'mean', beta=1.0):
    """
    Custom implementation of Smooth L1 Loss
    
    Args:
        predictions: Tensor of predictions
        targets: Tensor of target values
        beta: The threshold parameter that determines the switch point (default: 1.0)
    
    Returns:
        The computed Smooth L1 Loss
    """
    diff = torch.abs(predictions.to(torch.float64) - targets.to(torch.float64))
    condition = diff < beta
    
    # Where diff < beta, use quadratic form
    quadratic = 0.5 * diff.pow(2) / beta
    
    # Where diff >= beta, use linear form
    linear = diff - 0.5 * beta
    
    # Combine the two parts based on the condition
    loss = torch.where(condition, quadratic, linear)
    
    # Return loss
    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)
    elif reduction == "none":
        loss = loss
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}")
    return loss

def stable_huber_loss(predictions, targets, reduction: str = 'mean', delta=1.0):
    diff = torch.abs(predictions.to(torch.float64) - targets.to(torch.float64))
    abs_error = torch.abs(diff)
    
    # For small errors (≤ delta): use squared error (L2)
    quadratic = 0.5 * diff.pow(2)
    
    # For large errors (> delta): use modified absolute error (L1)
    linear = delta * (abs_error - 0.5 * delta)
    
    # Combine both parts
    loss = torch.where(abs_error <= delta, quadratic, linear)
    
    # Return loss
    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)
    elif reduction == "none":
        loss = loss
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}")
    return loss

def stable_l1_loss(predictions, targets, reduction: str = 'mean'):
    loss = torch.abs(predictions.to(torch.float64) - targets.to(torch.float64))

    # Return loss
    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)
    elif reduction == "none":
        loss = loss
    else:
        raise ValueError(f"Unsupported reduction type: {reduction}")
    return loss

def conditional_loss(
    model_pred: torch.Tensor, 
    target: torch.Tensor, 
    loss_type: str, 
    reduction: str,
    huber_c: Optional[torch.Tensor] = None,
    scale: float = 1.0,
):
    model_pred = model_pred.to(torch.float64)
    target = target.to(torch.float64)

    if huber_c is not None and huber_c.numel() > 1:
        huber_c_reshaped = huber_c.view(-1, *([1] * (model_pred.ndim - 1)))
    else:
        huber_c_reshaped = huber_c

    if loss_type == "l2":
        loss = stable_mse_loss(model_pred, target, reduction="none")
    elif loss_type == "l1":
        loss = stable_l1_loss(model_pred, target, reduction="none")
    elif loss_type == "standard_pseudo_huber":
        loss = stable_pseudo_huber_loss(model_pred, target, delta=huber_c_reshaped, reduction="none")
    elif loss_type == "standard_huber":
        loss = stable_huber_loss(model_pred, target, reduction="none", delta=huber_c_reshaped)
    elif loss_type == "standard_smooth_l1":
        loss = stable_smooth_l1_loss(model_pred, target, reduction="none", beta=huber_c_reshaped)
    elif loss_type == "huber":
        loss = 2 * huber_c_reshaped * (torch.sqrt(((model_pred - target)**2) + huber_c_reshaped**2) - huber_c_reshaped)
    elif loss_type == "smooth_l1":
        loss = 2 * (torch.sqrt(((model_pred - target)**2) + huber_c_reshaped**2) - huber_c_reshaped)
    elif loss_type == "x_sigmoid":
        loss = x_sigmoid_loss(model_pred, target, reduction="none")
    elif loss_type == "log_cosh":
        loss = stable_log_cosh_loss(model_pred, target, reduction="none")
    elif loss_type == "squared_logarithmic":
        loss = stable_msle_loss(model_pred, target, reduction="none")
    elif loss_type == "soft_welsch":
        loss = soft_welsch_loss(model_pred, target, reduction="none", delta=huber_c_reshaped, scale=scale)
    elif loss_type == "scaled_quadratic":
        loss = scaled_quadratic_loss(model_pred, target, reduction="none", delta=huber_c_reshaped)
    elif loss_type == "standard_deviation_loss":
        loss = standard_deviation_loss(model_pred, target, reduction="none")
    elif loss_type == "psnr_loss":
        # Bound the log at a perfect prediction instead of returning -inf.
        loss = 10 * torch.log10((model_pred-target).square().mean().clamp_min(1e-12))
    elif loss_type == "geman_mcclure_loss":
        loss = kornia.losses.geman_mcclure_loss(model_pred, target)
    elif loss_type == "smooth_l2_log":
        loss = smooth_l2_log_loss(model_pred, target, reduction="none", delta=huber_c_reshaped)
    elif loss_type == "mse_pyramid_2d":
        loss = mse_pyramid_loss_2d_non_reduced(model_pred, target, levels=5)
    else:
        raise NotImplementedError(f"Unsupported Loss Type: {loss_type}")
    
    if reduction == "mean":
        loss = torch.mean(loss)
    elif reduction == "sum":
        loss = torch.sum(loss)
    return loss

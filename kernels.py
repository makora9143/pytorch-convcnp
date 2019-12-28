import math
import torch


def eq_kernel(x, length_scale, output_scale, jitter=1e-8):
    r"""
    Args:
        x (Tensor): [num_points, x_dim]
        length_scale (Tensor): [y_dim, x_dim]
        output_scale (Tensor): [y_dim]
        jitter (int): for stability
    """
    num_points = x.size(0)

    x1 = x.unsqueeze(0)  # [1, num_points, x_dim]
    x2 = x.unsqueeze(1)  # [num_points, 1, x_dim]

    diff = x1 - x2  # [num_points, num_points, x_dim]

    # (x1 - x2)^2 / ll^2
    norm = diff[None, :, :, :].div(length_scale[:, None, None, :]).pow(2).sum(-1).clamp(0)  # [y_dim, num_points, num_points]
    # norm.clamp_(0)

    covariance = torch.exp(-0.5 * norm)  # [y_dim, num_points, num_points]

    scaled_covariance = output_scale.pow(2)[:, None, None] * covariance  # [y_dim, num_points, num_points]

    scaled_covariance = scaled_covariance + jitter * torch.eye(num_points)

    return scaled_covariance


def matern_kernel(x, length_scale, output_scale, jitter=1e-8):
    r"""
    Args:
        x (Tensor): [num_points, x_dim]
        length_scale (Tensor): [y_dim, x_dim]
        output_scale (Tensor): [y_dim]
        jitter (int): for stability
    """
    num_points = x.size(0)

    x1 = x.unsqueeze(0)  # [1, num_points, x_dim]
    x2 = x.unsqueeze(1)  # [num_points, 1, x_dim]

    diff = x1 - x2
    distance = (diff[None, :, :, :] / length_scale[:, None, None, :]).pow(2).sum(-1).clamp_(min=1e-30).sqrt_()  # [y_dim, num_points, num_points]

    exp_component = torch.exp(-math.sqrt(2.5 * 2) * distance)

    constant_component = (math.sqrt(5) * distance).add(1).add(5.0 / 3.0 * distance ** 2)
    covariance = constant_component * exp_component
    scaled_covariance = output_scale.pow(2)[:, None, None] * covariance  # [y_dim, num_points, num_points]
    scaled_covariance += jitter * torch.eye(num_points)
    return scaled_covariance


def periodic_kernel(x, length_scale, output_scale, jitter=1e-8):
    r"""
    Args:
        x (Tensor): [num_points, x_dim]
        length_scale (Tensor): [y_dim, x_dim]
        output_scale (Tensor): [y_dim]
        jitter (int): for stability
    """
    num_points = x.size(0)

    x1 = x.unsqueeze(0)  # [1, num_points, x_dim]
    x2 = x.unsqueeze(1)  # [num_points, 1, x_dim]

    diff = x1 - x2
    diff = (diff[None, :, :, :] / length_scale[:, None, None, :]).pow(2).sum(-1).clamp_(min=1e-30).sqrt_()  # [y_dim, num_points, num_points]

    covariance = torch.sin(diff.mul(math.pi)).pow(2).mul(-2 / length_scale).exp_()
    scaled_covariance = output_scale.pow(2)[:, None, None] * covariance  # [y_dim, num_points, num_points]
    scaled_covariance += jitter * torch.eye(num_points)
    return scaled_covariance

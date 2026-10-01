"""
D-FINE: Redefine Regression Task of DETRs as Fine-grained Distribution Refinement
Copyright (c) 2024 The D-FINE Authors. All Rights Reserved.
---------------------------------------------------------------------------------
Modified from RT-DETR (https://github.com/lyuwenyu/RT-DETR)
Copyright (c) 2023 lyuwenyu. All Rights Reserved.
"""

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def inverse_sigmoid(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    x = x.clip(min=0.0, max=1.0)
    return torch.log(x.clip(min=eps) / (1 - x).clip(min=eps))


def bias_init_with_prob(prior_prob=0.01):
    """initialize conv/fc bias value according to a given probability value."""
    bias_init = float(-math.log((1 - prior_prob) / prior_prob))
    return bias_init


def grid_sample_bilinear(im: torch.Tensor, grid: torch.Tensor) -> torch.Tensor:
    """Same as F.grid_sample(im, grid, mode="bilinear", padding_mode="zeros", align_corners=False),
    built from Floor/Pad/GatherElements so it exports to ONNX opsets below 16.

    Args:
        im (Tensor): [n, c, h, w]
        grid (Tensor): [n, gh, gw, 2], xy in [-1, 1]

    Returns:
        output (Tensor): [n, c, gh, gw]
    """
    n, c, h, w = im.shape
    _, gh, gw, _ = grid.shape

    x = ((grid[..., 0] + 1) * w - 1) / 2
    y = ((grid[..., 1] + 1) * h - 1) / 2
    x0 = x.floor()
    y0 = y.floor()

    wa = ((x0 + 1 - x) * (y0 + 1 - y)).unsqueeze(1)
    wb = ((x0 + 1 - x) * (y - y0)).unsqueeze(1)
    wc = ((x - x0) * (y0 + 1 - y)).unsqueeze(1)
    wd = ((x - x0) * (y - y0)).unsqueeze(1)

    # zero border of 1 pixel - any corner outside the image is clamped onto it and reads 0
    im_padded = F.pad(im, [1, 1, 1, 1]).reshape(n, c, -1)
    padded_w = w + 2
    # cast to int before building the flat index - in fp16 (e.g. TensorRT fp16) y * w + x
    # exceeds 2048 and loses integer precision
    x1 = (x0 + 2).clamp(0, padded_w - 1).to(torch.int64)
    x0 = (x0 + 1).clamp(0, padded_w - 1).to(torch.int64)
    y1 = (y0 + 2).clamp(0, h + 1).to(torch.int64)
    y0 = (y0 + 1).clamp(0, h + 1).to(torch.int64)

    def gather(yy, xx):
        idx = (yy * padded_w + xx).reshape(n, 1, -1).expand(-1, c, -1)
        return torch.gather(im_padded, 2, idx).reshape(n, c, gh, gw)

    return (
        gather(y0, x0) * wa + gather(y1, x0) * wb + gather(y0, x1) * wc + gather(y1, x1) * wd
    )


def deformable_attention_core_func(
    value, value_spatial_shapes, sampling_locations, attention_weights
):
    """
    Args:
        value (Tensor): [bs, value_length, n_head, c]
        value_spatial_shapes (Tensor|List): [n_levels, 2]
        value_level_start_index (Tensor|List): [n_levels]
        sampling_locations (Tensor): [bs, query_length, n_head, n_levels, n_points, 2]
        attention_weights (Tensor): [bs, query_length, n_head, n_levels, n_points]

    Returns:
        output (Tensor): [bs, Length_{query}, C]
    """
    bs, _, n_head, c = value.shape
    _, Len_q, _, n_levels, n_points, _ = sampling_locations.shape

    split_shape = [h * w for h, w in value_spatial_shapes]
    value_list = value.split(split_shape, dim=1)
    sampling_grids = 2 * sampling_locations - 1
    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        # N_, H_*W_, M_, D_ -> N_, H_*W_, M_*D_ -> N_, M_*D_, H_*W_ -> N_*M_, D_, H_, W_
        value_l_ = value_list[level].flatten(2).permute(0, 2, 1).reshape(bs * n_head, c, h, w)
        # N_, Lq_, M_, P_, 2 -> N_, M_, Lq_, P_, 2 -> N_*M_, Lq_, P_, 2
        sampling_grid_l_ = sampling_grids[:, :, :, level].permute(0, 2, 1, 3, 4).flatten(0, 1)
        # N_*M_, D_, Lq_, P_
        sampling_value_l_ = F.grid_sample(
            value_l_, sampling_grid_l_, mode="bilinear", padding_mode="zeros", align_corners=False
        )
        sampling_value_list.append(sampling_value_l_)
    # (N_, Lq_, M_, L_, P_) -> (N_, M_, Lq_, L_, P_) -> (N_*M_, 1, Lq_, L_*P_)
    attention_weights = attention_weights.permute(0, 2, 1, 3, 4).reshape(
        bs * n_head, 1, Len_q, n_levels * n_points
    )
    output = (
        (torch.stack(sampling_value_list, dim=-2).flatten(-2) * attention_weights)
        .sum(-1)
        .reshape(bs, n_head * c, Len_q)
    )

    return output.permute(0, 2, 1)


def deformable_attention_core_func_v2(
    value: torch.Tensor,
    value_spatial_shapes,
    sampling_locations: torch.Tensor,
    attention_weights: torch.Tensor,
    num_points_list: List[int],
    method="default",
):
    """
    Args:
        value (Tensor): [bs, value_length, n_head, c]
        value_spatial_shapes (Tensor|List): [n_levels, 2]
        value_level_start_index (Tensor|List): [n_levels]
        sampling_locations (Tensor): [bs, query_length, n_head, n_levels * n_points, 2]
        attention_weights (Tensor): [bs, query_length, n_head, n_levels * n_points]

    Returns:
        output (Tensor): [bs, Length_{query}, C]
    """
    bs, n_head, c, _ = value[0].shape
    _, Len_q, _, _, _ = sampling_locations.shape

    # sampling_offsets [8, 480, 8, 12, 2]
    if method in ("default", "gather"):
        sampling_grids = 2 * sampling_locations - 1

    elif method == "discrete":
        sampling_grids = sampling_locations

    sampling_grids = sampling_grids.permute(0, 2, 1, 3, 4).flatten(0, 1)
    sampling_locations_list = sampling_grids.split(num_points_list, dim=-2)

    sampling_value_list = []
    for level, (h, w) in enumerate(value_spatial_shapes):
        value_l = value[level].reshape(bs * n_head, c, h, w)
        sampling_grid_l: torch.Tensor = sampling_locations_list[level]

        if method == "default":
            sampling_value_l = F.grid_sample(
                value_l, sampling_grid_l, mode="bilinear", padding_mode="zeros", align_corners=False
            )

        elif method == "gather":
            sampling_value_l = grid_sample_bilinear(value_l, sampling_grid_l)

        elif method == "discrete":
            # n * m, seq, n, 2
            # clamp before the int64 cast - ONNX Clip only accepts integer inputs from opset 12
            # FIX ME? for rectangle input
            sampling_coord = (
                (sampling_grid_l * torch.tensor([[w, h]], device=value_l.device) + 0.5)
                .clamp(0, h - 1)
                .to(torch.int64)
            )
            sampling_coord = sampling_coord.reshape(bs * n_head, Len_q * num_points_list[level], 2)

            s_idx = (
                torch.arange(sampling_coord.shape[0], device=value_l.device)
                .unsqueeze(-1)
                .repeat(1, sampling_coord.shape[1])
            )
            sampling_value_l: torch.Tensor = value_l[
                s_idx, :, sampling_coord[..., 1], sampling_coord[..., 0]
            ]  # n l c

            sampling_value_l = sampling_value_l.permute(0, 2, 1).reshape(
                bs * n_head, c, Len_q, num_points_list[level]
            )

        sampling_value_list.append(sampling_value_l)

    attn_weights = attention_weights.permute(0, 2, 1, 3).reshape(
        bs * n_head, 1, Len_q, sum(num_points_list)
    )
    weighted_sample_locs = torch.concat(sampling_value_list, dim=-1) * attn_weights
    output = weighted_sample_locs.sum(-1).reshape(bs, n_head * c, Len_q)

    return output.permute(0, 2, 1)


def get_activation(act: str, inpace: bool = True):
    """get activation"""
    if act is None:
        return nn.Identity()

    elif isinstance(act, nn.Module):
        return act

    act = act.lower()

    if act == "silu" or act == "swish":
        m = nn.SiLU()

    elif act == "relu":
        m = nn.ReLU()

    elif act == "leaky_relu":
        m = nn.LeakyReLU()

    elif act == "silu":
        m = nn.SiLU()

    elif act == "gelu":
        m = nn.GELU()

    elif act == "hardsigmoid":
        m = nn.Hardsigmoid()

    else:
        raise RuntimeError("")

    if hasattr(m, "inplace"):
        m.inplace = inpace

    return m

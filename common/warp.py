import numpy as np
import cv2
from scipy.ndimage import map_coordinates
from scipy.interpolate import griddata

def forward_warp_bilinear(image, flow, flow_mask=None):
    height, width = flow.shape[:2]
    output = np.zeros_like(image, dtype=np.float32)
    weight_map = np.zeros((height, width), dtype=np.float32)

    grid_y, grid_x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    
    if flow_mask is None:
        dst_x = (grid_x + flow[:, :, 0]).flatten()
        dst_y = (grid_y + flow[:, :, 1]).flatten()
        src_pixels = image.reshape(-1, 3).astype(np.float32)
    else:
        dst_x = (grid_x + flow[:, :, 0])[flow_mask].flatten()
        dst_y = (grid_y + flow[:, :, 1])[flow_mask].flatten()
        src_pixels = image[flow_mask].reshape(-1, 3).astype(np.float32)
        
    x0 = np.floor(dst_x).astype(np.int32)
    y0 = np.floor(dst_y).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    wx = dst_x - x0
    wy = dst_y - y0

    wa = (1 - wx) * (1 - wy)
    wb = wx * (1 - wy)
    wc = (1 - wx) * wy
    wd = wx * wy

    weights = [wa, wb, wc, wd]
    coords = [
        (x0, y0),
        (x1, y0),
        (x0, y1),
        (x1, y1)
    ]

    for (x, y), w in zip(coords, weights):
        mask = (x >= 0) & (x < width) & (y >= 0) & (y < height)
        x_m = x[mask]
        y_m = y[mask]
        w_m = w[mask]
        pixels = src_pixels[mask]

        idx = np.ravel_multi_index((y_m, x_m), (height, width))
        np.add.at(output.reshape(-1, 3), idx, pixels * w_m[:, None])
        np.add.at(weight_map.reshape(-1), idx, w_m)

    warp_valid = weight_map != 0

    weight_map = weight_map[:, :, None]
    weight_map[weight_map == 0] = 1.0
    output /= weight_map

    return np.clip(output, 0, 255).astype(np.uint8), warp_valid

def forward_displacement_interpolation(image, flow, flow_mask):
    height, width = flow.shape[:2]

    # Source pixel grid
    grid_y, grid_x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')

    # Filter based on mask
    src_x = grid_x[flow_mask].flatten()
    src_y = grid_y[flow_mask].flatten()
    disp_x = flow[..., 0][flow_mask].flatten()
    disp_y = flow[..., 1][flow_mask].flatten()

    # Compute destination subpixel locations
    dst_x = src_x + disp_x
    dst_y = src_y + disp_y
    displaced_coords = np.stack([dst_y, dst_x], axis=-1)

    # Target grid (integer pixel locations at time t+1)
    tgt_y, tgt_x = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
    target_points = np.stack([tgt_y.ravel(), tgt_x.ravel()], axis=-1)

    # Interpolate u, v displacements to target grid
    interp_u = griddata(displaced_coords, disp_x, target_points, method='linear', fill_value=np.nan)
    interp_v = griddata(displaced_coords, disp_y, target_points, method='linear', fill_value=np.nan)

    interp_u = interp_u.reshape((height, width))
    interp_v = interp_v.reshape((height, width))

    # Create output mask: True where interpolation succeeded
    output_mask = ~np.isnan(interp_u) & ~np.isnan(interp_v)

    # Backward-warp coordinates
    back_x = np.zeros_like(tgt_x, dtype=np.float32)
    back_y = np.zeros_like(tgt_y, dtype=np.float32)
    back_x[output_mask] = tgt_x[output_mask] - interp_u[output_mask]
    back_y[output_mask] = tgt_y[output_mask] - interp_v[output_mask]

    back_x = np.clip(back_x, 0, width - 1)
    back_y = np.clip(back_y, 0, height - 1)

    # Initialize output image
    output = np.zeros_like(image, dtype=np.float32)

    # Apply bicubic interpolation for each channel
    for c in range(image.shape[2]):
        sampled = map_coordinates(image[..., c], [back_y, back_x], order=3, mode='constant', cval=0)
        output[..., c][output_mask] = sampled[output_mask]

    output = np.clip(output, 0, 255).astype(np.uint8)
    return output, output_mask
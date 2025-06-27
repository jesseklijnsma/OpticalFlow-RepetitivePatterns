import numpy as np

def reconstruction_error(
    gt_frame: np.ndarray,
    pred_frame: np.ndarray,
    gt_mask: np.ndarray = None,
    ):
    
    err = np.linalg.norm(gt_frame.astype(np.float32) - pred_frame.astype(np.float32), axis=-1)
    if gt_mask is not None:
        err = err[gt_mask]
    return err.mean()
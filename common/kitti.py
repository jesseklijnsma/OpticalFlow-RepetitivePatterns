import cv2
import numpy as np

def load_kitti_flow(filename):
    flow_png = cv2.imread(filename, -1)  # shape: H x W x 3, uint16
    if flow_png is None or flow_png.ndim != 3 or flow_png.shape[2] != 3:
        raise ValueError("Invalid flow file: {}".format(filename))

    valid = flow_png[:, :, 0] > 0  # mask
    u = (flow_png[:, :, 2].astype(np.float32) - 2**15) / 64.0
    v = (flow_png[:, :, 1].astype(np.float32) - 2**15) / 64.0
    flow = np.stack((u, v), axis=-1)  # shape: H x W x 2

    return flow, valid

def save_kitti_flow(filename, flow, valid):
    
    fx = (flow[:, :, 0] * 64.0) + 2**15
    fy = (flow[:, :, 1] * 64.0) + 2**15
    
    valid_mask = valid.astype(np.uint16) 
    
    flow_png = np.stack((valid, fy * valid_mask, fx * valid_mask), axis=-1).astype(np.uint16)
    cv2.imwrite(filename, flow_png)
import numpy as np 

def forward_warp(im0, im1, disp0, disp1, calib, x_pos, y_pos):
    """
    Forward warp images im0 and im1 to a new position using their disparity maps disp0 and disp1
    and calibration data. 

    Args:
        im0 : Left view of the scene 
        im1 : Right view of the scene
        disp0 : Left view ground truth disparity 
        disp1 : Right view ground truth disparity
        calib : Calibration information as a dictionary 
        x_pos: New x-position to reproject to (0.0 = im0(left), 1.0 = im1(right))
        y_pos: New y-position to reproject to (multiple of baseline)

    Returns:
        output_img: The reprojected view at the new position.
    """
    
    output_img = np.zeros_like(im0)
    h, w = im0.shape[:2]
    z_buffer = np.full((h, w), np.inf, dtype=np.float32) # Z-buffer for depth handling

    for src_index, (img, disp) in enumerate([(im0, disp0), (im1, disp1)]):
        depth = (calib['baseline'] * calib['cam0'][0,0]) / (disp + calib['doffs'])
        for y in range(h): 
            for x in range(w): 
                d = disp[y, x]
                if not np.isfinite(d): # skip invalid disparities
                    continue
                new_x = int(x - disp[y,x] * (x_pos - src_index))
                new_y = int(y - y_pos * (disp[y,x] + calib['doffs']))
                if 0 <= new_x < w and 0 <= new_y < h and depth[y, x] < z_buffer[new_y, new_x]:
                    output_img[new_y, new_x] = img[y, x]
                    z_buffer[new_y, new_x] = depth[y, x]
    
    return output_img

# TODO: fill holes from disocclusion & y-axis shifts 
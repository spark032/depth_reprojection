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

    # Forward warp each source image to the new position 
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
    
    # Fill holes created from x-shift (linear interpolation)
    for y in range(h): 
        x = 0
        while x < w: 
            if output_img[y, x].sum() == 0: # start of a hole
                # find left valid pixel 
                # TODO: need to check for cases where leftmost or rightmost pixel is a hole 
                # -> in this case, I can just copy the nearest valid pixel 
                left_x = x - 1
                if left_x < 0 or output_img[y, left_x].sum() == 0:
                    x += 1
                    continue 
                
                # find right valid pixel 
                right_x = x
                while right_x < w and output_img[y, right_x].sum() == 0:
                    right_x += 1
                if right_x == w: 
                    break

                # linear interpolate across the gap
                left_color = output_img[y, left_x].astype(np.float32)
                right_color = output_img[y, right_x].astype(np.float32)
                gap_size = right_x - left_x

                for i, fill_x in enumerate(range(left_x + 1, right_x)):
                    t = (i + 1) / gap_size
                    output_img[y, fill_x] = (left_color * (1 - t) + right_color * t).astype(np.uint8)
                x = right_x
            else:
                x += 1

    # Fill holes created from y-shift (copy nearest valid pixel)
    if y_pos != 0:
        for x in range(w):
            col = output_img[:, x]
            valid_mask = (col.sum(axis=1) > 0)
            
            if not valid_mask.any():
                continue
            
            # Find first and last valid row in this column
            valid_rows = np.where(valid_mask)[0]
            first_valid = valid_rows[0]
            last_valid = valid_rows[-1]
            
            # Fill above first valid (y_pos < 0)
            output_img[:first_valid, x] = output_img[first_valid, x]
            
            # Fill below last valid (y_pos > 0)
            output_img[last_valid+1:, x] = output_img[last_valid, x]

    return output_img
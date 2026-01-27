import numpy as np
import cv2 

def forward_warp(im0, im1, disp0, disp1, calib, x_pos):
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

    Returns:
        output_img: The reprojected view at the new position.
    """
    
    # Use original images when at exact camera positions
    if x_pos == 0:
        return im0.copy()
    elif x_pos == 1:
        return im1.copy()
    
    h, w = im0.shape[:2]
    
    # Accumulation buffers for bilinear splatting
    color_accum = np.zeros((h, w, 3), dtype=np.float64)
    weight_accum = np.zeros((h, w), dtype=np.float64)
    z_buffer = np.full((h, w), np.inf, dtype=np.float32)

    baseline = calib['baseline']
    focal = calib['cam0'][0, 0]
    doffs = calib['doffs']

    # Forward warp with bilinear splatting
    for src_index, (img, disp) in enumerate([(im0, disp0), (im1, disp1)]):
        depth = (baseline * focal) / (disp + doffs)
        for y in range(h):
            for x in range(w):
                d = disp[y, x]
                if not np.isfinite(d):
                    continue
                
                # Compute floating-point destination coordinates
                new_x_f = x - d * (x_pos - src_index)
                new_y_f = y  # No y-shift
                
                # Get integer and fractional parts
                x0 = int(np.floor(new_x_f))
                y0 = int(np.floor(new_y_f))
                fx = new_x_f - x0
                fy = new_y_f - y0
                
                pixel_depth = depth[y, x]
                pixel_color = img[y, x].astype(np.float64)
                
                # Splat to 2x2 neighborhood with bilinear weights
                for dy in range(2):
                    for dx in range(2):
                        px, py = x0 + dx, y0 + dy
                        if 0 <= px < w and 0 <= py < h:
                            # Bilinear weight
                            wx = fx if dx == 1 else (1 - fx)
                            wy = fy if dy == 1 else (1 - fy)
                            weight = wx * wy
                            
                            # Only splat if this pixel is closer (z-buffer test)
                            if pixel_depth < z_buffer[py, px] + 1.0:  # Small tolerance for blending
                                if pixel_depth < z_buffer[py, px]:
                                    # New closest - reset accumulation
                                    color_accum[py, px] = pixel_color * weight
                                    weight_accum[py, px] = weight
                                    z_buffer[py, px] = pixel_depth
                                else:
                                    # Similar depth - accumulate
                                    color_accum[py, px] += pixel_color * weight
                                    weight_accum[py, px] += weight

    # Normalize accumulated colors
    valid_mask = weight_accum > 0
    output_img = np.zeros_like(im0)
    output_img[valid_mask] = (color_accum[valid_mask] / weight_accum[valid_mask, np.newaxis]).astype(np.uint8)

    # Fill x-shift holes: per row, extend edges and linearly interpolate gaps
    for y in range(h):
        valid = np.where(output_img[y].sum(axis=1) > 0)[0]
        if len(valid) == 0:
            continue
        output_img[y, :valid[0]] = output_img[y, valid[0]]
        output_img[y, valid[-1] + 1:] = output_img[y, valid[-1]]
        for i in range(len(valid) - 1):
            left_x, right_x = valid[i], valid[i + 1]
            if right_x - left_x <= 1:
                continue
            left_c = output_img[y, left_x].astype(np.float32)
            right_c = output_img[y, right_x].astype(np.float32)
            for j, fill_x in enumerate(range(left_x + 1, right_x)):
                t = (j + 1) / (right_x - left_x)
                output_img[y, fill_x] = (left_c * (1 - t) + right_c * t).astype(np.uint8)

    # Apply median blur to reduce noise
    output_img = cv2.medianBlur(output_img, 7)

    return output_img


def create_video(im0, im1, disp0, disp1, calib, output_path, start_x=-0.3, end_x=1.3, step=0.1, fps=10):
    """
    Create a video by reprojecting views from start_x to end_x.

    Args:
        im0 : Left view of the scene
        im1 : Right view of the scene
        disp0 : Left view ground truth disparity
        disp1 : Right view ground truth disparity
        calib : Calibration information as a dictionary
        output_path : Path to save the output video (e.g., 'output.mp4')
        start_x : Starting x position (default -0.3)
        end_x : Ending x position (default 1.3)
        step : Step size for x position (default 0.1)
        fps : Frames per second for the video (default 10)

    Returns:
        None
    """
    h, w = im0.shape[:2]
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (w, h))
    
    # Generate frames
    x_positions = np.arange(start_x, end_x + step/2, step)  # +step/2 to include end_x
    
    for i, x_pos in enumerate(x_positions):
        print(f"Rendering frame {i+1}/{len(x_positions)}: x_pos={x_pos:.2f}")
        frame = forward_warp(im0, im1, disp0, disp1, calib, x_pos)
        # Convert RGB to BGR for OpenCV
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        video_writer.write(frame_bgr)
    
    video_writer.release()
    print(f"Video saved to {output_path}")
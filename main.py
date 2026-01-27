import argparse 
import sys
import cv2
from pathlib import Path

from load_data import load_scene
from forward_warp import forward_warp, create_video

def main(argv): 
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-d', '--dataset', type=Path, help='Directory with images and calibration info', required=True)
    p.add_argument('-x', '--x-pos', type=float, default=0.0, help='New x-position to reproject to', required=False) # 0.0 = im0(left), 1.0 = im1(right)
    p.add_argument('-v', '--video', type=bool, default=False, help='If set, saves a video of a reproject from position -1.0 to 2.0', required=False)
    args = p.parse_args(argv)

    if args.dataset is None: 
        p.error('--dataset is required')
    
    if args.x_pos is None: 
        p.error('x-position is required')

    im0, im1, disp0, disp1, calib = load_scene(args.dataset)

    if args.video: 
        create_video(im0, im1, disp0, disp1, calib, f'results/{args.dataset.name}.mp4')
    else: 
        output_img = forward_warp(im0, im1, disp0, disp1, calib, args.x_pos)
        cv2.imwrite(f'results/{args.dataset.name}_x{args.x_pos}.png', cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    main(sys.argv[1:])
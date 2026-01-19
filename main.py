import argparse 
import sys
import cv2
from pathlib import Path

from load_data import load_scene
from forward_warp import forward_warp

def main(argv): 
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('-d', '--dataset', type=Path, help='Directory with images and calibration info', required=True)
    p.add_argument('-x', '--x-pos', type=float, help='New x-position to reproject to', required=True) # 0.0 = im0(left), 1.0 = im1(right)
    p.add_argument('-y', '--y-pos', type=float, default=0.0, help='New y-position to reproject to', required=False) # multiple of baseline
    p.add_argument('-v', '--video', type=bool, default=False, help='If set, saves a video of a reproject from position -1.0 to 2.0', required=False) # TODO: implement later
    args = p.parse_args(argv)

    if args.dataset is None: 
        p.error('--dataset is required')
    
    if args.x_pos is None or args.y_pos is None: 
        p.error('both x-position and y-position are required')

    im0, im1, disp0, disp1, calib = load_scene(args.dataset)
    output_img = forward_warp(im0, im1, disp0, disp1, calib, args.x_pos, args.y_pos)
    cv2.imwrite(f'results/{args.dataset.name}_x{args.x_pos}_y{args.y_pos}.png', cv2.cvtColor(output_img, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    main(sys.argv[1:])
import cv2
import numpy as np

def load_pfm(path): 
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    img = np.where(np.isfinite(img), img, 0)
    return img 

def load_img(path):
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def parse_calib(path): 
    """
    Parses calibration file into a dictionary 

    Args: 
        path to calib.txt file 

    Returns: 
        dictionary with calibration parameters
    """
    calib = {} 
    with open(path, 'r') as f:
        for line in f:
            if '=' in line: 
                key, value = line.strip().split('=')
                key = key.strip()
                value = value.strip()
                if value.startswith('[') and value.endswith(']'):
                    value = value[1:-1]
                    rows = value.split(';')
                    matrix = []
                    for row in rows:
                        matrix.append([float(x) for x in row.split()])
                    calib[key] = np.array(matrix)
                else: 
                    try: 
                        calib[key] = float(value)
                    except ValueError:
                        calib[key] = value
    return calib

def load_scene(scene_path): 
    """
    Loads all the data from a Middlebury scene. 
    """
    im0 = load_img(scene_path / 'im0.png')
    im1 = load_img(scene_path / 'im1.png')
    disp0 = load_pfm(scene_path / 'disp0.pfm')
    disp1 = load_pfm(scene_path / 'disp1.pfm')
    calib = parse_calib(scene_path / 'calib.txt')

    return im0, im1, disp0, disp1, calib
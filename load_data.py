import cv2
import numpy as np
import re

def read_pfm(file):
    """
    from Microsoft AirSim
    https://github.com/microsoft/AirSim/blob/main/PythonClient/airsim/utils.py#L122
    """
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    header = str(bytes.decode(header, encoding='utf-8'))
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    temp_str = str(bytes.decode(file.readline(), encoding='utf-8'))
    dim_match = re.match(r'^(\d+)\s(\d+)\s$', temp_str)
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    file.close()
    
    return data, scale

def read_img(path):
    img = cv2.imread(str(path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def parse_calib(path): 
    """
    Parses calibration file into a dictionary 

    Args: 
        path: Path to calib.txt file 

    Returns: 
        Dictionary with calibration parameters
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
    im0 = read_img(scene_path / 'im0.png')
    im1 = read_img(scene_path / 'im1.png')
    disp0, _ = read_pfm(scene_path / 'disp0.pfm')
    disp1, _ = read_pfm(scene_path / 'disp1.pfm')
    calib = parse_calib(scene_path / 'calib.txt')

    # pfm stores images upside down, so need to flip to match image orientation
    disp0 = np.flipud(disp0)
    disp1 = np.flipud(disp1)

    return im0, im1, disp0, disp1, calib
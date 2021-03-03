import argparse
import glob
import os

import numpy as np
from imageio import imsave

from image import DepthImage, Image

import matplotlib.pyplot as plt

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate depth images from Cornell PCD files.')
    parser.add_argument('path', type=str, help='Path to Cornell Grasping Dataset')
    args = parser.parse_args()

    pcds = glob.glob(os.path.join(args.path, '*', 'pcd*[0-9]d.tiff'))
    pcds.sort()

    # min_v = float('inf')
    # max_v = -float('inf')
    for pcd in pcds:
        # 1. Load tiff
        di = DepthImage.from_tiff(pcd)

        # 1b. Rescale depth from [0-3](estimated 0-2.8) to [0-255]
        di.img = di.img * 85                    # Scale to 0-255
        di.img = np.clip(di.img, 0, 255)        # Clip to limit bw 0-255
        di.img = di.img.astype(np.uint8)        # Convert to uint8

        # 2. Load RGB
        rgb_name = pcd.replace('d.tiff', 'r.png')
        rgb = Image.from_file(rgb_name)

        # 3. Replace B(Blue) with D(Depth)
        rgb.img[:,:,2] = di

        # 4. Save back Image
        rgd_name = rgb_name.replace('r.png', 'z.png')
        imsave(rgd_name, rgb.img)
        
        print('Done: ', rgd_name)
        # # 5 debug. Get Max and min depths from all images
        # max_v = max( max_v, np.max(di))
        # min_v = min( min_v, np.min(di))
        ## Show Image
        # plt.imshow(rgb.img)
        # plt.show()
        
    # print('MAX: ', max_v, ' MIN: ', min_v)
# python convert_to_rgd.py /home/aby/Workspace/MTP/Datasets/Cornell/archive
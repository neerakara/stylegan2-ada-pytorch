import os
from pathlib import Path
import dataset_tool
import numpy as np
import PIL.Image

def load_real_images(nsamples):
    source_dir = '/data/vision/polina/users/nkarani/projects/anomaly/ideas/stylegan/datasets/fets_from_hdf5/'
    input_image_paths = [str(f) for f in sorted(Path(source_dir).rglob('*')) if dataset_tool.is_image_ext(f) and os.path.isfile(f)]
    
    nx, ny = np.array(PIL.Image.open(input_image_paths[0])).shape
    real_images = np.zeros(shape=(nsamples, nx, ny), dtype=np.float32)
    
    indices = np.random.randint(0, len(input_image_paths), nsamples)
    for i, idx in enumerate(indices):
        real_images[i] = convert_pil_to_arr(PIL.Image.open(input_image_paths[idx]))
    return real_images

def convert_pil_to_arr(img_pil):
    img = np.array(img_pil).astype(np.float32)
    img /= 127.5
    img -= 1.0
    return img
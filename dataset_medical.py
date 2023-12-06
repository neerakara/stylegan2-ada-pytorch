
import click
import h5py
from typing import Callable, Optional, Tuple, Union
import os
from PIL import Image
import numpy as np

#----------------------------------------------------------------------------
@click.command()
@click.pass_context
@click.option('--source', help='Location of the hdf5 file containing medical images', required=True, metavar='PATH')
@click.option('--dest', help='Location of the output directory where pngs should be stored', required=True, metavar='PATH')
def convert_dataset(ctx: click.Context,
                    source: str,
                    dest: str):
    """
    Convert a hdf5 file containing medical images to a folder containing individual images as pngs.
    This folder can then be passed as "source" to dataset_tool.py to create a dataset archive usable with StyleGAN2 ADA PyTorch.
    """
    if source == '':
        ctx.fail('--source filename or directory must not be an empty string')
    if dest == '':
        ctx.fail('--destination directory must not be an empty string')

    # Create the destination directory
    os.makedirs(dest, exist_ok=True)  # Create the directory if it doesn't exist

    # read hdf5 file
    data_medical = h5py.File(source, 'r')
    images = data_medical["images"]
    print(f"Read images of shape {images.shape} and dtype {images.dtype} from hdf5 file.")

    # Loop through each image and save as a grayscale PNG
    num_images = images.shape[2]
    for i in range(num_images):
        img = images[:, :, i]
        img_min = np.min(img)
        img_max = np.max(img)

        # Normalize pixel values between 0 and 255 (assuming the images contain float values)
        img_array = np.uint8((img - img_min) / (img_max - img_min) * 255.0)
        img_array = np.rot90(img_array, k = -1)

        # Convert numpy array to PIL Image
        img_png = Image.fromarray(img_array, mode='L')  # 'L' mode for grayscale

        # Save the image to the destination directory with a unique filename (e.g., image_0.png, image_1.png, ...)
        print(f"Saving file {i+1} out of {num_images}...")
        img_png.save(os.path.join(dest, f"image_{i}.png"))

    # close hdf5 file
    data_medical.close()

#----------------------------------------------------------------------------

if __name__ == "__main__":
    convert_dataset() # pylint: disable=no-value-for-parameter
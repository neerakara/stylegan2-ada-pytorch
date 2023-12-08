import io
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt

#----------------------------------------------------------------------------

def save_images(images,
                fname,
                nr = 6,
                nc = 8,
                sc = 10,
                cmaps = None,
                clims = None):
    """
    This function saves images to fname.
    """
    if not cmaps:
        cmaps = ['gray'] * len(images)
    if not clims:
        clims = [(-1.0, 1.0)] * len(images)
    plt.figure(figsize=(sc*nc, sc*nr))
    for r in range(nr):
        for c in range(nc):
            idx = r*nc + c
            plt.subplot(nr, nc, idx + 1, xticks=[], yticks=[])
            if cmaps[idx] is not None:
                plt.imshow(np.squeeze(images[idx]), cmap = cmaps[idx])
            else:
                plt.imshow(np.squeeze(images[idx]))
            if clims[idx] != None:
                cmin, cmax = clims[idx]
                plt.clim(cmin, cmax)
                plt.colorbar()
    
    if isinstance(fname, str):
        plt.savefig(fname, bbox_inches='tight', dpi=50)
    else: # fname is a in-memory IO biffer
        plt.savefig(fname, format='png', bbox_inches='tight', dpi=50)
        fname.seek(0)
    
    plt.close()

    return fname

#----------------------------------------------------------------------------

def prepare_video_frame(data):
    """
    This function prepares a video frame to be added to the video
    which shows the optimization process during
    joint image recovery and anomaly mask estimation.

    Args:
    - data: A list containing tuples of images, colormaps and clims

    Returns:
    - A NumPy array representing the video frame
    """
    images, cmaps, clims = zip(*data)
    buf = io.BytesIO()
    buf_images = save_images(images, buf, nr = 1, nc = len(images), cmaps = cmaps, clims = clims)
    video_frame = PIL.Image.open(buf_images).convert("RGB")

    # resize the video frame to be a multiple of 16
    size_x, size_y = video_frame.size
    size_x, size_y = 16 * (size_x // 16), 16 * (size_y // 16)
    video_frame = video_frame.resize(size=(size_x, size_y), resample=PIL.Image.Resampling.LANCZOS)
    
    return np.array(video_frame)

#----------------------------------------------------------------------------

def plot_losses_and_return_image(losses,
                                 step,
                                 size):
    """
    Plots losses until the current step and returns the resulting image.
    
    Args:
    - losses_data: Dictionary containing loss data for different types.
    - step: Current step.
    - image_size: Tuple specifying the size of the output image.
    
    Returns:
    - NumPy array representing the generated image.
    """
    # Get min-max values over all losses over all steps
    loss_min = min(min(loss_values) for loss_values in losses.values())
    loss_max = max(max(loss_values) for loss_values in losses.values())
    loss_min, loss_max = -0.1, 4.0 # hard-coding this for now
    
    # set line-styles for different losses
    line_styles = ['-', '--', '-.', ':']

    # Create a plot
    plt.figure(figsize=(20, 20))
    x_values = range(step+1)
    for i, (loss_type, loss_values) in enumerate(losses.items()):
        plt.plot(x_values, loss_values[:step+1], label=loss_type, linestyle = line_styles[i], linewidth = 10)
    plt.xlim(-1, len(loss_values))
    plt.ylim(loss_min, loss_max)
    plt.xticks(fontsize=100)
    plt.yticks(fontsize=100)
    plt.legend(fontsize=50)
    
    # Render the plot to an in-memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=50)
    buf.seek(0)
    
    # Convert the buffer content to a PIL image
    pil_image = PIL.Image.open(buf).convert("RGB")
    pil_image = pil_image.resize(size, PIL.Image.Resampling.LANCZOS)
    
    # Close the plot to release resources
    plt.close()

    return np.array(pil_image, dtype=np.uint8)
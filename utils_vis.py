import numpy as np
import matplotlib.pyplot as plt

# ==========================================================
# ==========================================================
def save_images(images, fname, nr = 6, nc = 8, sc = 6):
    
    plt.figure(figsize=(sc*nc, sc*nr))
    for r in range(nr):
        for c in range(nc):
            idx = r*nc + c
            plt.subplot(nr, nc, idx + 1, xticks=[], yticks=[])
            plt.imshow(np.squeeze(images[idx]), cmap = 'gray')
            plt.clim([-1.0, 1.0])
    plt.savefig(fname, bbox_inches='tight', dpi=50)
    plt.close()
    return 0
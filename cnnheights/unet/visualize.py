import matplotlib.pyplot as plt  # plotting tools
from matplotlib.patches import Polygon
def display_images(img, plot_path:str=None):
    """Display the given set of images, optionally with titles.
    images: array of image tensors in Batch * Height * Width * Channel format.
    titles: optional. A list of titles to display with each image.
    cmap: Optional. Color map to use. For example, "Blues".
    norm: Optional. A Normalize instance to map values to colors.
    interpolation: Optional. Image interpolation to use for display.
    """
    import matplotlib.pyplot as plt
    import os

    cols = img.shape[-1]
    rows = img.shape[0]

    fig, axs = plt.subplots(rows, cols, figsize=(14, 14 * rows // cols))
    for i in range(rows):
        if cols > 1:
            for j in range(cols):
                axs[i,j].axis('off')
                axs[i,j].imshow(img[i,...,j])
        else:
            axs[i].axis('off')
            axs[i].imshow(img[i,...]) 

    plt.tight_layout()
    if plot_path:
        if '.' not in plot_path:
            plot_path = os.path.join(plot_path, 'display_image_output.pdf')
        plt.savefig(plot_path)

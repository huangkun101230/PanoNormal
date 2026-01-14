import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"]="1"
import matplotlib.pyplot as plot
import numpy as np
import cv2

    
def save_norm_tensor_as_float(directory, filename, tensor):
    if not os.path.exists(directory):
        print("Given directory does not exist. Creating...")
        os.mkdir(directory)
    tensor = tensor.detach().cpu()
    array = tensor.squeeze(0).numpy()
    array = array.transpose(1, 2, 0)
    array = (array - np.min(array)) / (np.max(array) - np.min(array)) * 255

    filepath_exr = os.path.join(directory, filename + ".png")
    
    # array = cv2.cvtColor(array, cv2.COLOR_RGB2BGR)

    cv2.imwrite(filepath_exr, array)


def save_errormap_tensor_as_float(directory, filename, tensor):
    if not os.path.exists(directory):
        print("Given directory does not exist. Creating...")
        os.mkdir(directory)
    tensor = tensor.detach().cpu().numpy()[0]
    # array = tensor.squeeze(0).numpy()
    # array = tensor.transpose(1, 2, 0)
    filepath_exr = os.path.join(directory, filename + ".png")
    # cv2.imwrite(filepath_exr, array)
    plot.imsave(filepath_exr, tensor, cmap="Reds")
    # if (png):
    #     norm_array = (array - np.min(array)) / (np.max(array) - np.min(array)) * 255
    #     norm_array = norm_array.astype(np.uint8)
    #     filepath_png = os.path.join(directory, filename + ".png")
    #     cv2.imwrite(filepath_png, norm_array)
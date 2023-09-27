import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt

# Read the image in grayscale
# Modify your path!
img = cv.imread('', cv.IMREAD_GRAYSCALE)

# Decrease the grayscale values by reducing the pixel intensities
darkened_img = img * 0.8  # Adjust the multiplication factor (0.8 in this case)

# Keep pixel intensities within the valid range of 0 to 255
darkened_img = np.clip(darkened_img, 0, 255).astype(np.uint8)

# Perform histogram equalization
equ = cv.equalizeHist(darkened_img)

# Stacking original and equalized images side by side
res = np.hstack((darkened_img, equ))

# Save the result, or combine
# Modify your path!
cv.imwrite('', res)

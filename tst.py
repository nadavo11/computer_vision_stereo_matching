from solution import Solution
import numpy as np
import matplotlib.pyplot as plt

img_left = plt.imread('image_left.png')
img_right = plt.imread('image_right.png')

solution = Solution()
ssdd = solution.ssd_distance(img_left.astype(np.float64),
                             img_right.astype(np.float64),
                             win_size=3,
                             dsp_range=20)
plt.imshow(ssdd[:, :, 0], cmap='gray')
plt.show()

labling = solution.dp_labeling(ssdd, 0.5, 3.0)
plt.imshow(labling, cmap='gray')
plt.show()
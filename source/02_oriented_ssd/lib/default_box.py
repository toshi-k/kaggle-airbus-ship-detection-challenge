import math
import numpy as np
import pandas as pd


x_shifts = 0
y_shifts = 0
height_shifts = 3.5
width_shifts = 1.0
rotate_vars = np.linspace(-math.pi / 2.0, math.pi / 2, 5)[1:] - math.pi / 8

img_size = 768

dbox_params = np.stack(np.meshgrid(x_shifts, y_shifts, height_shifts, width_shifts, rotate_vars), -1).reshape(-1, 5)
dbox_params = pd.DataFrame(dbox_params,
                           columns=['x_shifts', 'y_shifts', 'height_shifts', 'width_shifts', 'rotate_vars'])
print(dbox_params)

# import numpy as np
from scipy.spatial.transform import Slerp, Rotation as R
print(R.from_euler('ZYX', [0, 30, 0], degrees=True).as_matrix())
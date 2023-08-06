import h5py
import numpy as np

f = h5py.File("build/particles_990.h5", "r")
# print(np.array(f["radius"]))
print(np.array(f["positions"]))
print(np.array(f["mass"]))
print(np.array(f["body_id"]))

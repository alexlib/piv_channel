import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import imageio.v3 as iio
import time
from openpiv import tools, pyprocess, validation, filters

print("Loading images...")
frame_a = iio.imread("B0001.tif")
frame_b = iio.imread("B0002.tif")

# Take a 1024x1024 crop
crop_a = frame_a[:1024, :1024].astype(np.int32)
crop_b = frame_b[:1024, :1024].astype(np.int32)

print(f"Crop shape: {crop_a.shape}")

print("Running PIV with use_vectorized=False...")
t0 = time.time()
u1, v1, s2n1 = pyprocess.extended_search_area_piv(
    crop_a, crop_b,
    window_size=64,
    overlap=32,
    dt=1.0,
    search_area_size=64,
    sig2noise_method='peak2mean',
    use_vectorized=False
)
t1 = time.time()
print(f"PIV with use_vectorized=False completed in {t1-t0:.4f} seconds.")

print("Running PIV with use_vectorized=True...")
t0 = time.time()
u2, v2, s2n2 = pyprocess.extended_search_area_piv(
    crop_a, crop_b,
    window_size=64,
    overlap=32,
    dt=1.0,
    search_area_size=64,
    sig2noise_method='peak2mean',
    use_vectorized=True
)
t1 = time.time()
print(f"PIV with use_vectorized=True completed in {t1-t0:.4f} seconds.")

print(f"u shape: {u1.shape}")
print(f"u diff: {np.nansum(np.abs(u1 - u2))}")

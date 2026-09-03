import imageio.v3 as iio
import numpy as np
import lvpyio as lv
import os

img = iio.imread("B0001.tif").astype(np.float32)
print("B0001.tif shape:", img.shape)

# Split vertically
frame_a = img[0:2048, :]
frame_b = img[2048:4096, :]

print("Frame A shape:", frame_a.shape)
print("Frame B shape:", frame_b.shape)

# Write as im7 buffer using lvpyio
try:
    os.makedirs("scratch", exist_ok=True)
    out_path = "scratch/B0001_split.im7"
    # To write a multi-frame buffer, pass a list of 2D arrays
    lv.write_buffer([frame_a, frame_b], out_path)
    print("Successfully wrote multi-frame im7 file using lvpyio!")
    
    # Read it back to verify
    buf = lv.read_buffer(out_path)
    print("Read back buffer:")
    print("  Frame count:", len(buf))
    print("  Frame 0 shape:", buf.as_masked_array(0).shape)
    print("  Frame 1 shape:", buf.as_masked_array(1).shape)
except Exception as e:
    print("Error with lvpyio read/write:", e)

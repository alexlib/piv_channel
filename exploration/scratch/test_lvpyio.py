import shutil
import os
import lvpyio

tif_file = "B0001.tif"
im7_file = "B0001.im7"

try:
    if os.path.exists(im7_file):
        os.remove(im7_file)
    shutil.copyfile(tif_file, im7_file)
    print(f"Copied {tif_file} to {im7_file}")
    
    buffer = lvpyio.read_buffer(im7_file)
    print("Successfully read B0001.im7 with lvpyio.read_buffer!")
    print("buffer type:", type(buffer))
    print("buffer count:", buffer.count)
    f0 = buffer.as_masked_array(0)
    print("Frame 0 shape:", f0.shape)
except Exception as e:
    print("Error reading renamed file:", e)
finally:
    if os.path.exists(im7_file):
        os.remove(im7_file)

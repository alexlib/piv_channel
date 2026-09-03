import imageio.v3 as iio
import numpy as np

for path in ["B0001.tif", "B0002.tif", "B0003.tif"]:
    print(f"\nInspecting {path}:")
    try:
        # Check length / number of frames using imageio
        props = iio.immeta(path)
        print("Properties:", props)
        
        # Read with imageio.v3
        img = iio.imread(path)
        print("iio.imread shape:", img.shape)
        
        # Read all frames if multi-page
        frames = iio.imread(path, index=None)
        print("All frames shape:", frames.shape)
    except Exception as e:
        print("Error:", e)

import imageio.v3 as iio

for name in ["B0001.tif", "B0002.tif", "B0003.tif"]:
    try:
        img = iio.imread(name)
        print(f"{name}: shape={img.shape}, dtype={img.dtype}, min={img.min()}, max={img.max()}")
    except Exception as e:
        print(f"Error reading {name}: {e}")

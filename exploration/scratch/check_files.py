import os
for root, dirs, files in os.walk("."):
    for file in files:
        if "B0001" in file:
            print(os.path.join(root, file))

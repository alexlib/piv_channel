import json

notebook_path = "manual/plots.ipynb"
with open(notebook_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print("Searching manual/plots.ipynb:")
keywords = ["lvpyio", "read_buffer", "read_set", "tif", "tiff", "im7", "vc7", "split", "top", "bottom", "half"]

for idx, cell in enumerate(nb["cells"]):
    if cell["cell_type"] == "code":
        source = "".join(cell["source"])
        source_lower = source.lower()
        matches = [k for k in keywords if k in source_lower]
        if matches:
            print(f"\n--- Code Cell {idx+1} (Matched: {matches}) ---")
            lines = source.split("\n")
            for line in lines[:15]:  # print first 15 lines of cell
                print(f"  {line}")
            if len(lines) > 15:
                print("  ...")

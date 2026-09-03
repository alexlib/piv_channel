html_path = "manual/README.html"
with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

print("Lines mentioning TIF/TIFF with context:")
for idx, line in enumerate(lines):
    if any(k in line.lower() for k in ["tif", "tiff"]):
        print(f"--- Line {idx+1} ---")
        start = max(0, idx - 3)
        end = min(len(lines), idx + 4)
        for i in range(start, end):
            prefix = "-> " if i == idx else "   "
            print(f"{prefix}{i+1}: {lines[i].strip()}")

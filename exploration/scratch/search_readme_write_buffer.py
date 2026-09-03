html_path = "manual/README.html"
with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
    lines = f.readlines()

print("Lines mentioning write_buffer with context:")
for idx, line in enumerate(lines):
    if "write_buffer" in line:
        print(f"--- Line {idx+1} ---")
        start = max(0, idx - 4)
        end = min(len(lines), idx + 10)
        for i in range(start, end):
            prefix = "-> " if i == idx else "   "
            print(f"{prefix}{i+1}: {lines[i].strip()}")

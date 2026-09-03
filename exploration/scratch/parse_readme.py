import re

html_path = "manual/README.html"
with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
    html_content = f.read()

# Strip HTML tags using simple regex
text = re.sub(r'<[^>]+>', ' ', html_content)
lines = text.split("\n")

print("Searching for keywords in README.html:")
keywords = ["tif", "tiff", "pair", "frame", "lvpyio", "lavision", "vertical", "horizontal", "half", "split", "top", "bottom"]

for line in lines:
    line_lower = line.lower()
    matches = [k for k in keywords if k in line_lower]
    if matches:
        cleaned_line = " ".join(line.split())
        if len(cleaned_line) > 10:
            print(f"- {cleaned_line[:120]} (Matched: {matches})")

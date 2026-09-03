import re

html_path = "manual/README.html"
with open(html_path, "r", encoding="utf-8", errors="ignore") as f:
    html_content = f.read()

# Let's find all instances of 'tif' or 'tiff' (case insensitive) and print their surrounding characters/context (150 chars before and after)
for match in re.finditer(r'tiff?', html_content, re.IGNORECASE):
    start = max(0, match.start() - 100)
    end = min(len(html_content), match.end() + 100)
    snippet = html_content[start:end]
    print(f"Match at {match.start()}:\n{snippet}\n" + "-"*50)

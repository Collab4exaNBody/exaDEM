from markdownify import markdownify as md

import subprocess

# 1. build mkdocs
subprocess.run(["mkdocs", "build"], check=True)

print("MkDocs build terminé")

with open("site/index.html") as f:
    html = f.read()

markdown = md(html)

with open("docs/api.md", "w") as f:
    f.write(markdown)

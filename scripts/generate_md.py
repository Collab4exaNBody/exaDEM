from markdownify import markdownify as md

with open("site/index.html") as f:
    html = f.read()

markdown = md(html)

with open("docs/api.md", "w") as f:
    f.write(markdown)

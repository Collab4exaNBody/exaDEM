import os
import subprocess
from pathlib import Path

BASE_DIRS = ["lib", "pre_processing", "post_processing"]
DOCS_API_DIR = Path("docs/api")
MKDOCS_NAV_FILE = Path("mkdocs_nav.yml")


# --------------------------------------------------
# 1. Scan des modules Python
# --------------------------------------------------
def scan_modules():
    structure = {}

    for base in BASE_DIRS:
        base_path = Path(base)
        modules = []

        for file in sorted(base_path.glob("*.py")):
            if file.name == "__init__.py":
                continue
            modules.append(file.stem)

        structure[base] = modules

    return structure


# --------------------------------------------------
# 2. Génération des .md mkdocstrings
# --------------------------------------------------
def generate_markdown(structure):
    for package, modules in structure.items():
        out_dir = DOCS_API_DIR / package
        out_dir.mkdir(parents=True, exist_ok=True)

        for mod in modules:
            md_file = out_dir / f"{mod}.md"
            full_module = f"{package}.{mod}"

            md_file.write_text(
                f"# {full_module}\n\n::: {full_module}\n",
                encoding="utf-8"
            )

            print(f"[OK] {md_file}")


# --------------------------------------------------
# 3. Génération automatique du nav mkdocs
# --------------------------------------------------
def generate_nav(structure):
    lines = []
    lines.append("nav:")
    lines.append("  - Home: index.md")
    lines.append("  - API:")

    for package, modules in structure.items():
        lines.append(f"      - {package.capitalize()}:")
        for mod in modules:
            path = f"api/{package}/{mod}.md"
            label = mod.replace("_", " ").title()
            lines.append(f"          - {label}: {path}")

    MKDOCS_NAV_FILE.write_text("\n".join(lines), encoding="utf-8")
    print(f"[OK] nav généré -> {MKDOCS_NAV_FILE}")


# --------------------------------------------------
# 4. Build mkdocs
# --------------------------------------------------
def build_docs():
    subprocess.run(["mkdocs", "build"], check=True)
    print("[OK] mkdocs build terminé")


# --------------------------------------------------
# MAIN
# --------------------------------------------------
def main():
    structure = scan_modules()
    generate_markdown(structure)
    generate_nav(structure)
    build_docs()


if __name__ == "__main__":
    main()

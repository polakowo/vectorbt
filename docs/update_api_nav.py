from pathlib import Path

from ruamel.yaml import YAML


def generate_nav_from_api(nav, root_dir=".", curr_dir="api"):
    root_dir = Path(root_dir)
    curr_dir = Path(curr_dir)
    full_path = root_dir / curr_dir

    nav.append(str(curr_dir / "index.md"))
    for p in sorted(full_path.iterdir()):
        if p.is_dir():
            sub_nav = []
            generate_nav_from_api(sub_nav, root_dir=root_dir, curr_dir=curr_dir / p.name)
            nav.append({p.name: sub_nav})
        elif p.name != "index.md":
            nav.append({p.stem: str(curr_dir / p.name)})


if __name__ == "__main__":
    mkdocs_path = Path("mkdocs.yml")

    yaml = YAML()
    yaml.indent(mapping=2, sequence=4, offset=2)
    yaml.width = 9999
    yaml.explicit_start = False
    yaml.explicit_end = False
    yaml.preserve_quotes = True

    data = yaml.load(mkdocs_path.read_text(encoding="utf-8"))
    if "nav" not in data or not isinstance(data["nav"], list):
        raise ValueError("mkdocs.yml has no top-level 'nav' list")

    api_nav = []
    generate_nav_from_api(api_nav, root_dir="docs")

    for item in data["nav"]:
        if isinstance(item, dict) and "API" in item:
            item["API"] = api_nav
            break
    else:
        raise ValueError("Couldn't find an 'API' section under nav")

    yaml.dump(data, mkdocs_path.open("w", encoding="utf-8"))

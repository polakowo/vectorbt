from pathlib import Path

import tomlkit


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


def nav_to_toml(nav):
    """Represent nested navigation with inline tables inside TOML arrays."""
    result = tomlkit.array().multiline(True)
    for item in nav:
        if isinstance(item, dict):
            entry = tomlkit.inline_table()
            for title, target in item.items():
                entry[title] = nav_to_toml(target) if isinstance(target, list) else target
            result.append(entry)
        else:
            result.append(item)
    return result


if __name__ == "__main__":
    config_path = Path(__file__).with_name("zensical.toml")
    data = tomlkit.parse(config_path.read_text(encoding="utf-8"))
    nav = data.get("project", {}).get("nav")
    if not isinstance(nav, list):
        raise ValueError("zensical.toml has no 'project.nav' list")

    api_nav = []
    generate_nav_from_api(api_nav, root_dir=config_path.parent / "docs")

    for item in nav:
        if isinstance(item, dict) and "API" in item:
            item["API"] = nav_to_toml(api_nav)
            break
    else:
        raise ValueError("Couldn't find an 'API' section under nav")

    config_path.write_text(tomlkit.dumps(data), encoding="utf-8")

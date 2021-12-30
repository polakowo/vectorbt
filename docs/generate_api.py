from functools import partial

from pdoc_to_md import generate_api, format_github_link


format_github_link = partial(format_github_link, user='polakowo', repo='vectorbt')

if __name__ == "__main__":
    generate_api(
        '../vectorbt',
        root_dir='docs',
        get_icon=lambda module: None,
        get_tags=lambda module: set(),
        format_github_link=format_github_link
    )

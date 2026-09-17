# Documentation

This directory contains documentation for vectorbt.

It is built using [Zensical](https://zensical.org/) and hosted on https://vectorbt.dev/.

## Build and preview

From the repository root, using Python 3.11 or newer:

```sh
python -m pip install -e ".[full,docs]"
cd docs
python generate_api.py
python update_api_nav.py
zensical build --strict
zensical serve
```

The editable install is important: the API generator must import this checkout of
`vectorbt`. Re-run both Python scripts after changing the API. Generated API
Markdown, the site output, and Zensical's cache are ignored by Git. The navigation
script updates the `API` section in `zensical.toml`; keep its generated entries out of
commits. GitHub Actions runs the same generation and strict-build steps and deploys
`site/` through GitHub Pages.

## Website configuration and content

- `zensical.toml`: Zensical configuration, navigation, Markdown extensions, analytics,
  and theme settings.
- `docs/`: Markdown pages and static assets, including interactive chart HTML.
- `generate_api.py` and `templates/markdown.mako`: API Markdown generation and
  formatting.
- `update_api_nav.py`: updates navigation from the generated API Markdown; run it
  separately after the generator.
- `overrides/main.html`: announcement, sponsor footer, social metadata, and icons.
- `docs/assets/stylesheets/extra.css`: styles for custom website components.

Run `zensical build --strict` before submitting website changes, and use
`zensical serve` to check affected pages and interactions in the browser.

The API generator can warn about dynamically generated classes and unresolved
docstring references. Review these separately from Zensical's strict build
validation.

## License

The code in this directory is licensed under the [GNU Affero General Public License v3.0 or later](https://github.com/polakowo/vectorbt/blob/master/docs/LICENSE.md).

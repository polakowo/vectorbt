---
title: Contributing
---

# Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

First, you need to install vectorbt from the repository:

```bash
pip uninstall vectorbt
git clone https://github.com/polakowo/vectorbt.git
cd vectorbt
pip install -e .
```

After making changes, make sure you did not break any functionality:

```bash
pytest
```

Make sure to update tests as appropriate.

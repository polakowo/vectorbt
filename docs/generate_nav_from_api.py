import yaml
from api_markdown import generate_nav_from_api

if __name__ == "__main__":
    nav = []
    generate_nav_from_api(nav, root_dir='docs')

    with open('nav.yml', 'w') as f:
        yaml.dump(nav, f, default_flow_style=False)

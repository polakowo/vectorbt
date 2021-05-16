from vectorbt import settings


# ############# settings.py ############# #

class TestSettings:
    def test_save_and_load(self, tmp_path):
        settings.set_theme('seaborn')
        settings.save(tmp_path / "settings")
        new_settings = settings.load(tmp_path / "settings")
        assert settings == new_settings
        assert settings.__dict__ == new_settings.__dict__

from vectorbt import settings


# ############# settings.py ############# #

class TestSettings:
    def test_save_and_load(self, tmp_path):
        initial_settings = dict()
        for k in settings.__all__:
            initial_settings[k] = getattr(settings, k)
        settings.save(tmp_path / "settings")
        settings.load(tmp_path / "settings")
        for k in settings.__all__:
            assert initial_settings[k] == getattr(settings, k)

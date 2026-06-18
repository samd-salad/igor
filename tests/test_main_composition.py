import importlib


def test_main_module_imports_cleanly(monkeypatch, tmp_path):
    """Building the app shouldn't crash. We don't actually start uvicorn."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    monkeypatch.setenv("HA_URL", "http://10.0.40.5:8123")
    monkeypatch.setenv("HA_TOKEN", "x")
    monkeypatch.setenv("BRAIN_DIR", str(tmp_path))

    # Force re-import so env vars apply at module load
    import sys
    for mod_name in list(sys.modules):
        if mod_name.startswith("server."):
            sys.modules.pop(mod_name, None)

    main = importlib.import_module("server.main")
    app = main.build()
    assert app is not None

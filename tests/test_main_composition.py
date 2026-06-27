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


def test_build_wires_hybrid_retrieval_when_embeddings_enabled(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_DIR", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x-test")
    monkeypatch.setenv("HA_TOKEN", "x-test")
    monkeypatch.delenv("IGOR_EMBEDDING_DISABLED", raising=False)

    captured_retrieval_class: list = []

    import server.main as m
    from server.cognition.services.conversation import Conversation
    original_init = Conversation.__init__

    def spy_init(self, **kwargs):
        captured_retrieval_class.append(type(kwargs["retrieval"]).__name__)
        original_init(self, **kwargs)

    monkeypatch.setattr(Conversation, "__init__", spy_init)
    m.build()

    assert captured_retrieval_class == ["HybridRetrieval"]


def test_build_wires_tag_retrieval_when_embeddings_disabled(tmp_path, monkeypatch):
    monkeypatch.setenv("BRAIN_DIR", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "x-test")
    monkeypatch.setenv("HA_TOKEN", "x-test")
    monkeypatch.setenv("IGOR_EMBEDDING_DISABLED", "1")

    captured_retrieval_class: list = []

    import server.main as m
    from server.cognition.services.conversation import Conversation
    original_init = Conversation.__init__

    def spy_init(self, **kwargs):
        captured_retrieval_class.append(type(kwargs["retrieval"]).__name__)
        original_init(self, **kwargs)

    monkeypatch.setattr(Conversation, "__init__", spy_init)
    m.build()

    assert captured_retrieval_class == ["TagRetrieval"]

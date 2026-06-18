"""Ports are Protocols (PEP 544). Verify shape, not behavior."""
from server.cognition.ports import persistence, retrieval, llm, tools, clock


def test_persistence_port_methods():
    assert hasattr(persistence.PersistencePort, "save_episode")
    assert hasattr(persistence.PersistencePort, "load_episode")
    assert hasattr(persistence.PersistencePort, "save_fact")
    assert hasattr(persistence.PersistencePort, "list_unconsolidated_episodes")


def test_retrieval_port_methods():
    assert hasattr(retrieval.RetrievalPort, "query")


def test_llm_port_methods():
    assert hasattr(llm.LLMPort, "chat")


def test_tool_executor_methods():
    assert hasattr(tools.ToolExecutorPort, "execute")
    assert hasattr(tools.ToolExecutorPort, "list_schemas")


def test_clock_methods():
    assert hasattr(clock.ClockPort, "now")

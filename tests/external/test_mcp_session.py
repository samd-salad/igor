import pytest

from server.external._internal.mcp_session import McpTool, _content_to_text


def test_mcp_tool_is_frozen_dataclass():
    t = McpTool(name="HassTurnOn", description="Turn something on",
                input_schema={"type": "object", "properties": {}})
    assert t.name == "HassTurnOn"
    with pytest.raises(Exception):
        t.name = "other"  # frozen


def test_content_to_text_concatenates_text_blocks():
    class _TB:
        type = "text"
        text = "Lights on."
    class _TB2:
        type = "text"
        text = " Done."
    out = _content_to_text([_TB(), _TB2()])
    assert out == "Lights on. Done."


def test_content_to_text_falls_back_to_str_for_unknown_block():
    class _Img:
        type = "image"
        def __str__(self): return "<image>"
    assert _content_to_text([_Img()]) == "<image>"


def test_content_to_text_handles_empty_list():
    assert _content_to_text([]) == ""

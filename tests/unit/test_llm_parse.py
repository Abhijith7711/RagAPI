import pytest

from app.services.llm import parse_llm_json


def test_parse_plain_json():
    result = parse_llm_json(
        '{"answer": "18 days", "found_in_documents": true, "supporting_chunk_ids": ["a:0"]}'
    )
    assert result.answer == "18 days"
    assert result.found_in_documents is True
    assert result.supporting_chunk_ids == ["a:0"]


def test_parse_fenced_json():
    raw = """Here you go
```json
{"answer": "unavailable", "found_in_documents": false, "supporting_chunk_ids": []}
```
"""
    result = parse_llm_json(raw)
    assert result.found_in_documents is False


def test_parse_invalid_json():
    with pytest.raises(ValueError):
        parse_llm_json("not json at all")

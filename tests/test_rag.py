import pytest
from rag_pipeline import query_rag

def test_query_rag():
    response = query_rag("Test query")  # Assume ingested doc
    assert isinstance(response, str) and len(response) > 0
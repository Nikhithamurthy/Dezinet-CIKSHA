from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from app.dependencies import limiter
from app.main import app

HEADERS = {"X-API-Key": "dev-key-123"}


@pytest.fixture
def client() -> Iterator[TestClient]:
    limiter.reset()
    with TestClient(app) as test_client:  # `with` runs the lifespan startup/shutdown
        yield test_client

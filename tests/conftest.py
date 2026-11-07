import pytest

from promptfence import Detector


@pytest.fixture(scope="session")
def detector() -> Detector:
    return Detector()


@pytest.fixture(scope="session")
def rules_only() -> Detector:
    return Detector(classifier=False)

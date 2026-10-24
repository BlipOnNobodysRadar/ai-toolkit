"""Create the original review's evidence destination without editing assertions."""
from pathlib import Path

import pytest


@pytest.fixture(scope='session', autouse=True)
def review_evidence_directory():
    (Path(__file__).resolve().parents[1] / 'evidence').mkdir(parents=True, exist_ok=True)

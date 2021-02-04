"""Tests for lowrank/__init__.py."""

import lowrank
import pytest
import time

NAMES = ["world", "github"]
IDS = NAMES


@pytest.mark.parametrize("name", NAMES, ids=IDS)
def test_hello(name):
    """Test hello function."""
    lowrank.hello(name)


@pytest.mark.expensive
@pytest.mark.parametrize("name", NAMES, ids=IDS)
def test_hello_expensive(name):
    """Expensive test of hello function. Will only be run on master ad development."""
    time.sleep(1)
    lowrank.hello(name)

"""Shared fixtures."""

from __future__ import annotations

import numpy as np
import pytest

from bora import Config


@pytest.fixture
def cfg() -> Config:
    """The default physical layer."""
    return Config()


@pytest.fixture
def fast_cfg() -> Config:
    """A smaller, faster configuration for tests that do not care about range."""
    return Config(n_fft=256, cp_len=64, chirp_len=1024, lead_silence=512)


@pytest.fixture
def rng() -> np.random.Generator:
    """A seeded generator, so a failure can be reproduced."""
    return np.random.default_rng(12345)


@pytest.fixture
def payload(rng: np.random.Generator) -> bytes:
    """Two kilobytes of incompressible noise."""
    return bytes(rng.integers(0, 256, 2_000, dtype=np.uint8))

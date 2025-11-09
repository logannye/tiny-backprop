from __future__ import annotations

import os
import random

import numpy as np

DEFAULT_SEED = int(os.getenv("TINY_BACKPROP_SEED", "1234"))


def pytest_configure(config) -> None:  # pylint: disable=unused-argument
    random.seed(DEFAULT_SEED)
    np.random.seed(DEFAULT_SEED)

    try:
        import torch

        torch.manual_seed(DEFAULT_SEED)
    except ModuleNotFoundError:
        pass

    try:
        from jax import random as jrandom

        # initialise a default key for any tests that rely on jax.random
        jrandom.PRNGKey(DEFAULT_SEED)
    except ModuleNotFoundError:
        pass


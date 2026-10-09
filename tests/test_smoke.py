import numpy as np

from mg2t.utils import set_seed


def test_seed_is_reproducible():
    set_seed(0)
    a = np.random.rand(3)
    set_seed(0)
    b = np.random.rand(3)
    assert np.allclose(a, b)

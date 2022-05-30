import os
import pytest
from adapters import CoordSystemVariant

def test_pick():
    spherical = CoordSystemVariant.pick('spherical')
    assert spherical is CoordSystemVariant.SPHERICAL

def test_str_method():
    spherical = CoordSystemVariant.pick('SpHericaL')
    actual = str(spherical)
    assert actual == 'Spherical'

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

import os
import pytest
from adapters import IntegratorVariant

def test_pick():
    actual = IntegratorVariant.pick('rk2 mp')
    assert actual is IntegratorVariant.RK2MP

def test_str_method():
    actual = IntegratorVariant.pick('rk3 Heun')
    actual = str(actual)
    assert actual == 'RK3 Heun'

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

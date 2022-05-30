import os
import pytest
from adapters import NumericalIntegration, IntegratorVariant
from collections import namedtuple

Stub = namedtuple('Stub', 'name fd_order real cfl_factor lapse_condition shift_condition')

class TestBuildFactoryMethod:

    @pytest.fixture
    def integrator(self):
        _integrator = NumericalIntegration.build()
        return _integrator

    @pytest.fixture
    def expected_attrs(self):
        stub = Stub(IntegratorVariant.RK4, 4, 'double', 0.5, 'OnePlusLog', "GammaDriving2ndOrder_Covariant")
        return stub

    def test_name_attr(self, integrator, expected_attrs):
        attr = 'name'
        actual, expected = map(lambda _: getattr(_, attr), (integrator, expected_attrs))
        assert actual == expected

    def test_fd_order_attr(self, integrator, expected_attrs):
        attr = 'fd_order'
        actual, expected = map(lambda _: getattr(_, attr), (integrator, expected_attrs))
        assert actual == expected

    def test_real_attr(self, integrator, expected_attrs):
        attr = 'real'
        actual, expected = map(lambda _: getattr(_, attr), (integrator, expected_attrs))
        assert actual == expected

    def test_cfl_factor_attr(self, integrator, expected_attrs):
        attr = 'cfl_factor'
        actual, expected = map(lambda _: getattr(_, attr), (integrator, expected_attrs))
        assert actual == expected

    def test_lapse_condition_attr(self, integrator, expected_attrs):
        attr = 'lapse_condition'
        actual, expected = map(lambda _: getattr(_, attr), (integrator, expected_attrs))
        assert actual == expected

    def test_shift_condition_attr(self, integrator, expected_attrs):
        attr = 'shift_condition'
        actual, expected = map(lambda _: getattr(_, attr), (integrator, expected_attrs))
        assert actual == expected

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

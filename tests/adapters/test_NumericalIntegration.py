import os
import pytest
from adapters import NumericalIntegration, IntegratorVariant
from collections import namedtuple
from unittest import mock

Stub = namedtuple('Stub', 'name fd_order real cfl_factor lapse_condition shift_condition')

class TestBuildFactoryMethod:

    @pytest.fixture
    def expected_rk_order(self):
        return 'Butcher_dict_call'

    @pytest.fixture
    def butcher_dict_stub(self, expected_rk_order, expected_attrs):
        with mock.patch.dict('nrpy_local.Butcher_dict', {str(expected_attrs.name): (None, expected_rk_order)}) as m:
            yield m

    @pytest.fixture
    def integrator(self, butcher_dict_stub):
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

    def test_rk_roder_attr(self, integrator, expected_rk_order):
        attr = 'rk_order'
        actual = getattr(integrator, attr)
        assert actual == expected_rk_order

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

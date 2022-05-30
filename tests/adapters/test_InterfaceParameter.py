import os
import pytest
from adapters import InterfaceParameter
from collections import namedtuple
from unittest import mock

Stub = namedtuple('Stub',  'parameter value representation')

class TestEmptyConstruction:

    @pytest.fixture
    def set_parval_spy(self, mocker):
        with mocker.patch('nrpy_local.par.set_parval_from_str') as spy:
            yield spy

    @pytest.fixture
    def parval_from_spy(self, mocker):
        with mocker.patch('nrpy_local.par.parval_from_str') as spy:
            yield spy

    @pytest.fixture
    def interface_parameter(self, set_parval_spy, parval_from_spy):
        _interface_parameter = InterfaceParameter()
        return _interface_parameter

    @pytest.fixture
    def expected_attrs(self):
        expected = Stub(None, None, None)
        return expected

    def test_parameter_attr(self, interface_parameter, expected_attrs):
        attr = 'parameter'
        acutal, expected = map(lambda _: getattr(_, attr), (interface_parameter, expected_attrs))
        assert acutal is expected

    def test_value_attr(self, interface_parameter, expected_attrs):
        attr = 'value'
        acutal, expected = map(lambda _: getattr(_, attr), (interface_parameter, expected_attrs))
        assert acutal is expected

    def test_representation_attr(self, interface_parameter, expected_attrs):
        attr = 'representation'
        acutal, expected = map(lambda _: getattr(_, attr), (interface_parameter, expected_attrs))
        assert acutal is expected

class TestBuildFactoryMethod:

    TC_INPUT = ('grid::DIM', 3)
    TC_REPRESENTATION = 'grid::DIM = 3'

    @pytest.fixture
    def set_parval_spy(self, mocker):
        with mocker.patch('nrpy_local.par.set_parval_from_str') as spy:
            yield spy

    @pytest.fixture
    def parval_from_spy(self, mocker):
        with mocker.patch('nrpy_local.par.parval_from_str', return_value=self.TC_REPRESENTATION) as spy:
            yield spy

    @pytest.fixture
    def interface_parameter(self, set_parval_spy, parval_from_spy):
        _interface_parameter = InterfaceParameter.build(*self.TC_INPUT)
        return _interface_parameter

    @pytest.fixture
    def expected_attrs(self):
        expected = Stub(*self.TC_INPUT, self.TC_REPRESENTATION)
        return expected

    def test_parameter_attr(self, interface_parameter, expected_attrs):
        attr = 'parameter'
        acutal, expected = map(lambda _: getattr(_, attr), (interface_parameter, expected_attrs))
        assert acutal == expected

    def test_value_attr(self, interface_parameter, expected_attrs):
        attr = 'value'
        acutal, expected = map(lambda _: getattr(_, attr), (interface_parameter, expected_attrs))
        assert acutal == expected

    def test_representation_attr(self, interface_parameter, expected_attrs):
        attr = 'representation'
        acutal, expected = map(lambda _: getattr(_, attr), (interface_parameter, expected_attrs))
        assert acutal == expected

    def test_as_string(self, interface_parameter):
        expected = self.TC_REPRESENTATION
        actual = interface_parameter.as_string()
        assert actual == expected

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

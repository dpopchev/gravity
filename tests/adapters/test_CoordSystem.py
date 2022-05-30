import os
import pytest
from adapters import CoordSystem, CoordSystemVariant
from collections import namedtuple

Stub = namedtuple('Stub', 'name domain_size sinh_width sinhv2_const_dr symtp_bscale')

class TestBuildSphericalFactoryMethod:

    @pytest.fixture
    def coord_system(self):
        _coord_system = CoordSystem.build_spherical()
        return _coord_system

    @pytest.fixture
    def expected_attrs(self):
        stub =Stub(CoordSystemVariant.SPHERICAL, 32, None, None, None)
        return stub

    def test_name_attr(self, coord_system, expected_attrs):
        attr = 'name'
        actual, expected = map(lambda _: getattr(_, attr), (coord_system, expected_attrs))
        assert actual == expected

    def test_domain_size_attr(self, coord_system, expected_attrs):
        attr = 'domain_size'
        actual, expected = map(lambda _: getattr(_, attr), (coord_system, expected_attrs))
        assert actual == expected

    def test_sinh_width_attr(self, coord_system, expected_attrs):
        attr = 'sinh_width'
        actual, expected = map(lambda _: getattr(_, attr), (coord_system, expected_attrs))
        assert actual == expected

    def test_sinhv2_const_dr_attr(self, coord_system, expected_attrs):
        attr = 'sinhv2_const_dr'
        actual, expected = map(lambda _: getattr(_, attr), (coord_system, expected_attrs))
        assert actual == expected

    def test_symtp_bscale_attr(self, coord_system, expected_attrs):
        attr = 'symtp_bscale'
        actual, expected = map(lambda _: getattr(_, attr), (coord_system, expected_attrs))
        assert actual == expected

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

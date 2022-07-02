import os
import pytest
from adapters import CoordSystem, CoordSystemVariant

@pytest.fixture
def make_mocked_coord_system(mocker):
    def factory(**kwargs):
        mocked = mocker.Mock()
        mocked.name = kwargs.get('name', None)
        mocked.domain_size = kwargs.get('domain_size', None)
        mocked.sinh_width = kwargs.get('sinh_width', None)
        mocked.sinhv2_const_dr = kwargs.get('sinhv2_const_dr', None)
        mocked.symtp_bscale = kwargs.get('symtp_bscale', None)

        return kwargs, mocked
    return factory

@pytest.fixture
def spherical_coord_system(make_mocked_coord_system):
    parameters, expected = make_mocked_coord_system(name=CoordSystemVariant.SPHERICAL, domain_size=32)
    return parameters, expected

class TestSphericalFactoryMethod:

    @pytest.fixture
    def coord_system(self, spherical_coord_system):
        parameters, expected = spherical_coord_system
        _coord_system = CoordSystem.build_spherical(parameters['domain_size'])
        return _coord_system, expected

    def test_name_attr_type(self, coord_system):
        actual, _ = coord_system
        attr = 'name'
        attr_value =getattr(actual, attr)
        assert isinstance(attr_value, CoordSystemVariant)

    def test_name_attr(self, coord_system):
        attr = 'name'
        actual, expected = (getattr(o, attr) for o in (coord_system))
        assert actual is expected

    def test_domain_size_attr(self, coord_system):
        attr = 'domain_size'
        actual, expected = (getattr(o, attr) for o in (coord_system))
        assert actual == expected

    def test_sinh_width_attr(self, coord_system):
        attr = 'sinh_width'
        actual, expected = (getattr(o, attr) for o in (coord_system))
        assert actual == expected

    def test_sinhv2_const_dr_attr(self, coord_system):
        attr = 'sinhv2_const_dr'
        actual, expected = (getattr(o, attr) for o in (coord_system))
        assert actual == expected

    def test_symtp_bscale_attr(self, coord_system):
        attr = 'symtp_bscale'
        actual, expected = (getattr(o, attr) for o in (coord_system))
        assert actual == expected

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

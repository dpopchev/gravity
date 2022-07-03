import pytest

from ccode_builders import build_scalar_field_initial_data_ccode_generator
from collections import namedtuple

MockFactoryParameters = namedtuple('MockFactoryParameters', ('args', 'kwargs'))

@pytest.fixture
def ccodes_dir(generic_ccodes_dir):
    _, _ccodes_dir = generic_ccodes_dir
    return _ccodes_dir

@pytest.fixture
def coord_system(spherical_coord_system):
    _, _coord_system = spherical_coord_system
    return _coord_system

@pytest.fixture
def make_sf_init_data_ccode_gen_builder(mocker):
    def factory(ccodes_dir, coord_system, **kwargs):
        outputfilename = kwargs.get('outputfilename', None)
        id_family = kwargs.get('id_family', None)
        pulse_amplitude = kwargs.get('pulse_amplitude', None)
        pulse_center = kwargs.get('pulse_center', None)
        pulse_width = kwargs.get('pulse_width', None)
        nr = kwargs.get('nr', None)
        rmax_coef = kwargs.get('rmax_coef', None)
        rmax = coord_system.domain_size*rmax_coef if rmax_coef is not None else None

        name = 'Scalar Field Initial Data Ccode Generator'
        callback = nrpy.sfid.ScalarField_InitialData

        mocked = mocker.Mock()
        mocked.name = name
        mocked.callback = callback
        mocked.args = (outputfilename, id_family, pulse_amplitude, pulse_center,
                pulse_width, nr, rmax)

        return {'name': name, 'callback': callback, 'args': args}, mocked
    return factory

@pytest.fixture
def ccode_generation(self):
    with mock.patch('nrpy_local.sfid') as m:
        m.__module__ = 'ScalarField.ScalarField_InitialData'
        m.__name__ = 'ScalarField_InitialData'
        yield m

@pytest.fixture
def generic_scalarf_field_init_data_ccode_generator(mocker, make_sf_init_data_ccode_gen_builder, ccodes_dir, coord_system):
    outputfilename  = ccodes_dir.make_under_outdir()
    id_family       = "Gaussian_pulse"
    pulse_amplitude = 0.4
    pulse_center    = 0
    pulse_width     = 1
    nr              = 30000
    rmax            = coord_system.domain_size*1.1
    return mocked.args, mocked

@pytest.mark.skip(reason='work in progress')
class TestDefaultBuilderScalarFieldInitData:

    @pytest.fixture
    def coord_system(self):
        _coord_system = mock.Mock()
        _coord_system.domain_size = 12
        return _coord_system

    @pytest.fixture
    def stub(self, ccodes_dir, coord_system):
        outputfilename  = ccodes_dir.make_under_outdir()
        id_family       = "Gaussian_pulse"
        pulse_amplitude = 0.4
        pulse_center    = 0
        pulse_width     = 1
        nr              = 30000
        rmax            = coord_system.domain_size*1.1

        name = 'Scalar Field Initial Data Ccode Generator'
        callback = 'ScalarField.ScalarField_InitialData.ScalarField_InitialData'
        args = (outputfilename, id_family, pulse_amplitude, pulse_center,
                pulse_width, nr, rmax )
        kwargs = { }

        _expected = ScalarFieldInitDataStub(name, callback, args, kwargs)
        return _expected


    @pytest.fixture
    def scalar_field_init_data(self, ccodes_dir, coord_system, ccode_generation):
        _scalar_field_init_data = build_scalar_field_initial_data_ccode_generator(ccodes_dir, coord_system)
        return _scalar_field_init_data

    def test_name_attr(self, scalar_field_init_data, stub):
        attr = 'name'
        actual, expected = (getattr(_, attr) for _ in (timestepping_ccode_generator, stub))
        assert actual == expected

    def test_callback_attr(self, timestepping_ccode_generator, stub):
        attr = 'callback'
        actual, expected = (getattr(_, attr) for _ in (timestepping_ccode_generator, stub))
        actual = '.'.join((actual.__module__, actual.__name__))
        assert actual == expected

    def test_args_attr(self, timestepping_ccode_generator, stub):
        attr = 'args'
        actual, expected = (getattr(_, attr) for _ in (timestepping_ccode_generator, stub))
        assert actual == expected

    def test_kwargs_attr(self, timestepping_ccode_generator, stub):
        attr = 'kwargs'
        actual, expected = (getattr(_, attr) for _ in (timestepping_ccode_generator, stub))
        assert actual == expected

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

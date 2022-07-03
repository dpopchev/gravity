import pytest

from ccode_builders import build_scalar_field_initial_data_ccode_generator

from tests.adapters import spherical_coord_system

@pytest.fixture
def make_mocked_scalar_field_init_data_ccode_generator(mocker):
    def factory(**kwargs):


@pytest.fixture
def ccode_generation(self):
    with mock.patch('nrpy_local.sfid') as m:
        m.__module__ = 'ScalarField.ScalarField_InitialData'
        m.__name__ = 'ScalarField_InitialData'
        yield m

@pytest.fixture
def make_mocked_scalar_field_init_data(mocker, ccode_generation, spherical_coord_system, ):
    def factory(**kwargs):
        outputfilename  = ccodes_dir.make_under_outdir()
        id_family       = "Gaussian_pulse"
        pulse_amplitude = 0.4
        pulse_center    = 0
        pulse_width     = 1
        nr              = 30000
        rmax            = coord_system.domain_size*1.1

        mocked = mocker.Mock()
        mocked.name = 'Scalar Field Initial Data Ccode Generator'
        mocked.callback = 'ScalarField.ScalarField_InitialData.ScalarField_InitialData'
        mocked.args = (outputfilename, id_family, pulse_amplitude, pulse_center, pulse_width, nr, rmax )
        mocked.kwargs = { }

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

    # class TestDoitMethod:
        
    #     @pytest.fixture
    #     def doit(self, timestepping_ccode_generator):
    #         timestepping_ccode_generator.doit()

    #     def test_make_under_root(self, doit, ccodes_dir):
    #         expected_destination = 'MoLtimestepping'
    #         ccodes_dir.make_under_root.assert_called_once_with(expected_destination)

    #     def test_callback_parameters(self, doit, c_code_generation, stub):
    #         c_code_generation.assert_called_once_with(*stub.args, **stub.kwargs)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

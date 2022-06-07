import pytest
from unittest import mock

from ccode_builders import build_timestepping_ccode_generator
from collections import namedtuple

TimesteppingCcodeGeneratorStub = namedtuple('TimesteppingCcodeGeneratorStub',
                                            'name callback args kwargs')

class TestDefaultBuilder:

    @pytest.fixture
    def ccodes_dir(self):
        _ccodes_dir = mock.Mock()
        _ccodes_dir.make_under_root = mock.Mock(return_value='make_under_root')
        return _ccodes_dir

    @pytest.fixture
    def numerical_integration(self):
        _numerical_integration = mock.Mock()
        _numerical_integration.rk_method = 'rk_method'
        return _numerical_integration

    @pytest.fixture
    def cmd_mkdir(self):
        with mock.patch('nrpy_local.cmd.mkdir') as m:
            yield m

    @pytest.fixture
    def stub(self, ccodes_dir,numerical_integration):
        name = 'Timestepping Ccode Generator'
        callback = 'MoLtimestepping.C_Code_Generation.MoL_C_Code_Generation'
        args = (numerical_integration.rk_method, )
        kwargs = {
            'RHS_string': "\n".join((
                'Ricci_eval(&rfmstruct, &params, RK_INPUT_GFS, auxevol_gfs); // DPythonMark',
                'rhs_eval(&rfmstruct, &params, auxevol_gfs, RK_INPUT_GFS, RK_OUTPUT_GFS); // DPythonMark',
            )),
            'post_RHS_string': "\n".join((
                'apply_bcs_curvilinear(&params, &bcstruct, NUM_EVOL_GFS, evol_gf_parity, RK_OUTPUT_GFS); // DPythonMark',
                'enforce_detgammahat_constraint(&rfmstruct, &params, RK_OUTPUT_GFS); // DPythonMark'
            )),
            'outdir': ccodes_dir.make_under_root()
        }

        _expected = TimesteppingCcodeGeneratorStub(name, callback, args, kwargs)
        return _expected

    @pytest.fixture
    def c_code_generation(self):
        with mock.patch('nrpy_local.MoL.MoL_C_Code_Generation') as m:
            m.__module__ = 'MoLtimestepping.C_Code_Generation'
            m.__name__ = 'MoL_C_Code_Generation'
            yield m

    @pytest.fixture
    def timestepping_ccode_generator(self, ccodes_dir, numerical_integration, c_code_generation):
        _timestepping_ccode_generator = build_timestepping_ccode_generator(ccodes_dir, numerical_integration)
        return _timestepping_ccode_generator

    def test_name_attr(self, timestepping_ccode_generator, stub):
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

    class TestDoitMethod:
        
        @pytest.fixture
        def doit(self, timestepping_ccode_generator):
            timestepping_ccode_generator.doit()

        def test_make_under_root(self, doit, ccodes_dir):
            expected_destination = 'MoLtimestepping'
            ccodes_dir.make_under_root.assert_called_once_with(expected_destination)

        def test_callback_parameters(self, doit, c_code_generation, stub):
            c_code_generation.assert_called_once_with(*stub.args, **stub.kwargs)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

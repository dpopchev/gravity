import pytest
from unittest import mock

from ccode_builders import build_moltimestepping_c_code_generation


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
    def builder(self, ccodes_dir, numerical_integration):

        def _builder():
            build_moltimestepping_c_code_generation(ccodes_dir, numerical_integration)
            return

        return _builder

    @pytest.fixture
    def c_code_generation(self):
        with mock.patch('nrpy_local.MoL.MoL_C_Code_Generation') as m:
            m.return_value = 0
            yield m

    @pytest.fixture
    def cmd_mkdir(self):
        with mock.patch('nrpy_local.cmd.mkdir') as m:
            yield m

    @pytest.fixture
    def c_code_generation_parameters(self, ccodes_dir, numerical_integration):
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
        return args, kwargs

    def test_c_code_generation_called(self, builder, c_code_generation,
                                      c_code_generation_parameters):
        builder()
        args, kwargs = c_code_generation_parameters
        c_code_generation.assert_called_once_with(*args, **kwargs)

    def test_cmd_mkdir_called(self, builder, c_code_generation, ccodes_dir):
        builder()
        hardcoded_destination = 'MoLtimestepping'
        ccodes_dir.make_under_root.assert_called_once_with(hardcoded_destination)

if __name__ == '__main__':
    pytest.main([__file__, '-vv'])

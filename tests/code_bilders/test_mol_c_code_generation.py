import pytest

from timestepping_code import build_moltimestepping_c_code_generation

class TestDefaultBuilder:

    @pytest.fixture
    def numerical_integration(self, mocker):
        _numerical_integration = mocker.Mock()
        _numerical_integration.name = 'IntegrationMethodName'
        return _numerical_integration

    @pytest.fixture
    def ccodes_dir(self, mocker):
        _ccodes_dir = mocker.Mock()
        _ccodes_dir.make_under_root = mocker.Mock(return_value='MoLtimestepping/')
        return _ccodes_dir

    @pytest.fixture
    def builder(self):
        return build_moltimestepping_c_code_generation

    @pytest.fixture
    def product(self, builder):
        result = builder()
        return result

    @pytest.fixture
    def name(self):
        return 'moltimestepping_c_code_generation'

    @pytest.fixture
    def args(self):
        return ()

    @pytest.fixture
    def rhs_string(self):
        return "\n".join((
            '// DP mark begin',
            'Ricci_eval(&rfmstruct, &params, RK_INPUT_GFS, auxevol_gfs);',
            'rhs_eval(&rfmstruct, &params, auxevol_gfs, RK_INPUT_GFS, RK_OUTPUT_GFS);',
            '// DP mark end',
        ))
    @pytest.fixture
    def post_rhs_string(self):
        return "\n".join((
            '// DP mark begin',
            'apply_bcs_curvilinear(&params, &bcstruct, NUM_EVOL_GFS, evol_gf_parity, RK_OUTPUT_GFS);',
            'enforce_detgammahat_constraint(&rfmstruct, &params,RK_OUTPUT_GFS);',
            '// DP mark end',
        ))

    @pytest.fixture
    def outdir(self):
        return 'some/dir/path'

    @pytest.fixture
    def kwargs(self, rhs_string, post_rhs_string, outdir):
        return {
            'RHS_string': rhs_string,
            'post_RHS_string': post_rhs_string,
            'outdir': outdir
        }

    @pytest.fixture
    def callback(self, mocker):
        mocker.patch('nrpy_local.MoLtimestepping.C_Code_Generation')

    class TestBuilderProduct:

        def test_name_attr(self, product, name):
            attr = 'name'
            actual, expected = getattr(product, attr), name
            assert actual == expected

        def test_args_attr(self, product, args):
            attr = 'args'
            actual, expected = getattr(product, attr), args
            assert actual == expected

        def test_kwargs_attr(self, product, kwargs):
            attr = 'kwargs'
            actual, expected = getattr(product, attr), kwargs
            assert actual == expected

        def test_callback_attr(self, product, callback):
            attr = 'callback'
            actual, expected = getattr(product, attr), callback
            assert actual is expected

        def test_doit_method_called_once(self, product, callback):
            product.doit()
            callback.assert_called_once()

        def test_doit_method_arguments(self, product, callback, args, kwargs):
            product.doit()
            callback.assert_called_with(*args, **kwargs)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])

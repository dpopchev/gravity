import pytest

@pytest.fixture
def ricci_eval():
    return 'Ricci_eval(&rfmstruct, &params, RK_INPUT_GFS, auxevol_gfs);'

@pytest.fixture
def rhs_eval():
    return 'rhs_eval(&rfmstruct, &params, auxevol_gfs, RK_INPUT_GFS, RK_OUTPUT_GFS);'

@pytest.fixture
def apply_bcs_curvilinear():
    return 'apply_bcs_curvilinear(&params, &bcstruct, NUM_EVOL_GFS, evol_gf_parity, RK_OUTPUT_GFS);'

@pytest.fixture
def enforce_detgammahat_constraint():
    return 'enforce_detgammahat_constraint(&rfmstruct, &params, RK_OUTPUT_GFS);'

if __name__ == '__main__':
    pytest.main([__file__, '-v'])

import nrpy_local as nrpy

from adapters import CcodePrototypeArgument, CcodePrototype

def build_rhs_string():
    funcname = 'Ricci_eval'
    parameters = (
        CcodePrototypeArgument('rfmstruct', 1),
        CcodePrototypeArgument('params', 1),
        CcodePrototypeArgument('RK_INPUT_GFS', 0),
        CcodePrototypeArgument('auxevol_gfs', 0),
    )
    ricci_eval = CcodePrototype(funcname, parameters)

    funcname = 'rhs_eval'
    parameters = (
        CcodePrototypeArgument('rfmstruct', 1),
        CcodePrototypeArgument('params', 1),
        CcodePrototypeArgument('auxevol_gfs', 0),
        CcodePrototypeArgument('RK_INPUT_GFS', 0),
        CcodePrototypeArgument('RK_OUTPUT_GFS', 0),
    )
    rhs_eval = CcodePrototype(funcname, parameters)

    joined = '\n'.join((str(_) for _ in (ricci_eval, rhs_eval)))
    return joined

def build_post_rhs_string():
    funcname = 'apply_bcs_curvilinear'
    parameters = (
        CcodePrototypeArgument('params', 1),
        CcodePrototypeArgument('bcstruct', 1),
        CcodePrototypeArgument('NUM_EVOL_GFS', 0),
        CcodePrototypeArgument('evol_gf_parity', 0),
        CcodePrototypeArgument('RK_OUTPUT_GFS', 0),
    )
    apply_bcs_curvilinear = CcodePrototype(funcname, parameters)

    funcname = 'enforce_detgammahat_constraint'
    parameters = (
        CcodePrototypeArgument('rfmstruct', 1),
        CcodePrototypeArgument('params', 1),
        CcodePrototypeArgument('RK_OUTPUT_GFS', 0),
    )
    enforce_detgammahat_constraint = CcodePrototype(funcname, parameters)

    joined = '\n'.join((str(_) for _ in (apply_bcs_curvilinear, enforce_detgammahat_constraint )))
    return joined

def build_moltimestepping_c_code_generation(ccodes_dir, numerical_integration,
                                            destination='MoLtimestepping'):
    _destination = ccodes_dir.make_under_root(destination)
    parameters = {
        'RHS_string': build_rhs_string(),
        'post_RHS_string': build_post_rhs_string(),
        'outdir': _destination
    }
    nrpy.MoL.MoL_C_Code_Generation(numerical_integration.rk_method, **parameters)

    return

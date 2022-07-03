import nrpy_local as nrpy

from adapters import CcodePrototypeArgument, CcodePrototype, NrpyAttrWrapper

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

def build_timestepping_ccode_generator(ccodes_dir,
                                       numerical_integration,
                                       destination='MoLtimestepping'):
    _destination = ccodes_dir.make_under_root(destination)
    args = (str(numerical_integration.name), )
    parameters = {
        'RHS_string': build_rhs_string(),
        'post_RHS_string': build_post_rhs_string(),
        'outdir': _destination
    }
    timestepping_ccode_generator = NrpyAttrWrapper(
        name = 'Timestepping Ccode Generator',
        callback = nrpy.MoL.MoL_C_Code_Generation,
        args = args,
        kwargs = parameters
    )
    return timestepping_ccode_generator

def build_scalar_field_initial_data_ccode_generator(ccodes_dir,
                                                    coord_system,
                                                    **kwargs):
    destination = kwargs.get('destination', 'SFID.txt')
    outputfilename  = ccodes_dir.make_under_outdir(destination, is_dir=False)

    id_family = kwargs.get('id_family', "Gaussian_pulse")
    pulse_amplitude = kwargs.get('pulse_amplitude', 0.4)
    pulse_center = kwargs.get('pulse_center', 0)
    pulse_width = kwargs.get('pulse_width', 1)
    nr = kwargs.get('nr', 30000)

    rmax_coef = kwargs.get('rmax_coef', 1.1)
    rmax = coord_system.domain_size*rmax_coef

    name = 'Scalar Field Initial Data Ccode Generator'
    callback = nrpy.sfid.ScalarField_InitialData

    args = (outputfilename, id_family, pulse_amplitude, pulse_center,
            pulse_width, nr, rmax)
    scalar_field_init_data = NrpyAttrWrapper(
        name = name,
        callback = callback,
        args = args,
        kwargs = {},
    )
    return scalar_field_init_data

def build_param_funcs_basic_defines_scalar_field(ccodes_dir):
    name = 'C code param functions and basic defines'
    callback = nrpy.sfid.NRPy_param_funcs_register_C_functions_and_NRPy_basic_defines
    kwargs = {'Ccodesdir': ccodes_dir.root}

    param_funcs_basic_defines_scalar_field = NrpyAttrWrapper(
        name = name,
        callback = callback,
        kwargs = kwargs,
        args = tuple()
    )

    return param_funcs_basic_defines_scalar_field

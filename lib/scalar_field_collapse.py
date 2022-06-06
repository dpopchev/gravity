import adapters
import nrpy_local as nrpy

def build_timestepping(ccodes_dir, numerical_integration):
    rfmstruct = adapters.CcodePrototypeArgument('rfmstruct', 1)
    params = adapters.CcodePrototypeArgument('params', 1)
    rk_input_gfs = adapters.CcodePrototypeArgument('RK_INPUT_GFS', 0)
    rk_output_gfs = adapters.CcodePrototypeArgument('RK_OUTPUT_GFS', 0)
    auxevol_gfs = adapters.CcodePrototypeArgument('auxevol_gfs', 0)
    bcstruct = adapters.CcodePrototypeArgument('bcstruct', 1)

    ricc_eval = adapters.CcodePrototype(
        name = 'Ricci_eval',
        arguments = (rfmstruct, params, rk_input_gfs, auxevol_gfs)
    )
    rhs_eval = adapters.CcodePrototype(
        name = 'rhs_eval',
        arguments = (rfmstruct, params, auxevol_gfs, rk_input_gfs, rk_output_gfs)
    )

def build():
    ccodes_dir = adapters.CcodesDir.build()
    dim = adapters.InterfaceParameter.build('grid::DIM', 3)
    coord_system = adapters.CoordSystem.build_spherical()
    numerical_integration = adapters.NumericalIntegration.build()

import os
import adapters
import nrpy_local as nrpy
from ccode_builders import build_timestepping_ccode_generator
from ccode_builders import build_scalar_field_initial_data_ccode_generator
from ccode_builders import build_param_funcs_basic_defines_scalar_field
from ccode_builders import build_converter_adm_bssn_init_data

def build():
    ccodes_dir = adapters.CcodesDir.build()
    dim = adapters.InterfaceParameter.build('grid::DIM', 3)
    coord_system = adapters.CoordSystem.build_spherical()
    numerical_integration = adapters.NumericalIntegration.build()

    timestepping_ccode_generator = build_timestepping_ccode_generator(ccodes_dir, numerical_integration)
    timestepping_ccode_generator.doit()

    reference_metric = adapters.InterfaceParameter.build("reference_metric::CoordSystem", str(coord_system.name))
    nrpy.rfm.reference_metric()

    finite_difference = adapters.InterfaceParameter.build('finite_difference::FD_CENTDERIVS_ORDER', numerical_integration.fd_order)

    simd_src = os.path.join('../nrpytutorial/SIMD/', 'SIMD_intrinsics.h')
    simd_dst = ccodes_dir.make_under_root('SIMD')
    nrpy.shutil.copy(simd_src,simd_dst)

    indexedexp = adapters.InterfaceParameter.build("indexedexp::symmetry_axes", "12")

    find_timestep_header = ccodes_dir.make_under_root('find_timestep.h', is_dir=False)
    nrpy.rfm.out_timestep_func_to_file(find_timestep_header)

    scalar_field_initial_data = build_scalar_field_initial_data_ccode_generator(ccodes_dir, coord_system)
    scalar_field_initial_data.doit()

    param_funcs_basic_defines_scalar_field = build_param_funcs_basic_defines_scalar_field(ccodes_dir)
    param_funcs_basic_defines_scalar_field.doit()

    converter_adm_bssn_init_data = build_converter_adm_bssn_init_data(ccodes_dir, coord_system)
    converter_adm_bssn_init_data.doit()

    lapse_evolution_option = adapters.InterfaceParameter.build('BSSN.BSSN_gauge_RHSs::LapseEvolutionOption', numerical_integration.lapse_condition)
    shift_evolution_option = adapters.InterfaceParameter.build('BSSN.BSSN_gauge_RHSs::ShiftEvolutionOption', numerical_integration.shift_condition)

    print("Generating symbolic expressions for BSSN RHSs...")

    # Enable rfm_precompute infrastructure, which results in
    #   BSSN RHSs that are free of transcendental functions,
    #   even in curvilinear coordinates, so long as
    #   ConformalFactor is set to "W" (default).
    enable_rfm_precompute = adapters.InterfaceParameter.build("reference_metric::enable_rfm_precompute","True")
    rfm_files = ccodes_dir.make_under_root('rfm_files')

    # Evaluate BSSN + BSSN gauge RHSs with rfm_precompute enabled:
    rfm_precompute_Ccode_outdir = adapters.InterfaceParameter.build("reference_metric::rfm_precompute_Ccode_outdir", rfm_files)
    leave_ricci_symbolic = adapters.InterfaceParameter.build("BSSN.BSSN_quantities::LeaveRicciSymbolic","True")
    nrpy.rhs.BSSN_RHSs()

    # Evaluate the Scalar Field RHSs
    nrpy.sfrhs.ScalarField_RHSs()

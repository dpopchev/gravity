import os
import adapters
import nrpy_local as nrpy
from ccode_builders import build_timestepping_ccode_generator
from ccode_builders import build_scalar_field_initial_data_ccode_generator
from ccode_builders import build_param_funcs_basic_defines_scalar_field

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

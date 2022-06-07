import adapters
import nrpy_local as nrpy
from ccode_builders import build_timestepping_ccode_generator

def build():
    ccodes_dir = adapters.CcodesDir.build()
    dim = adapters.InterfaceParameter.build('grid::DIM', 3)
    coord_system = adapters.CoordSystem.build_spherical()
    numerical_integration = adapters.NumericalIntegration.build()

    timestepping_ccode_generator = build_timestepping_ccode_generator(ccodes_dir, numerical_integration)
    timestepping_ccode_generator.doit()

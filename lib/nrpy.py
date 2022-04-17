from outputC import lhrh,outputC,outCfunction  # NRPy+: Core C code output module
import NRPy_param_funcs as par   # NRPy+: Parameter interface
import sympy as sp               # SymPy: The Python computer algebra package upon which NRPy+ depends
import finite_difference as fin  # NRPy+: Finite difference C code generation module
import grid as gri               # NRPy+: Functions having to do with numerical grids
import indexedexp as ixp         # NRPy+: Symbolic indexed expression (e.g., tensors, vectors, etc.) support
import reference_metric as rfm   # NRPy+: Reference metric support
import cmdline_helper as cmd     # NRPy+: Multi-platform Python command-line interface
import shutil, os, sys           # Standard Python modules for multiplatform OS-level functions
import MoLtimestepping.C_Code_Generation as MoL
from MoLtimestepping.RK_Butcher_Table_Dictionary import Butcher_dict
import ScalarField.ScalarField_InitialData as sfid

from dataclasses import dataclass, InitVar

CCODESDIR = os.path.join("ccodesdir_default")

@dataclass
class CodesDir:
    root: os.PathLike = CCODESDIR
    output: os.PathLike = None
    outputdir: InitVar[str] = 'output'

    def __post_init__(self, outputdir):
        if self.output is None:
            self.output = os.path.join(self.root, outputdir)

    def clean(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def build(self):
        self.clean()
        for directory in (self.root, self.output):
            cmd.mkdir(directory)

@dataclass
class SpatialDimension:
    value: int = 3
    dim: str = None
    ccodesdir: CodesDir = None

    def build_grid(self):
        par.set_parval_from_str('grid::DIM', self.value)
        self.dim = par.parval_from_str('grid::DIM')

    def build_find_timestep(self):
        target = os.path.join(self.ccodesdir.root, 'find_timestep.h')
        rfm.out_timestep_func_to_file(target)

@dataclass
class CoordSystem:
    coord_system: str = 'Spherical'
    domain_size: float = 32
    sinh_width: float = 0.2
    sinhv2_const_dr: float =0.05
    symtp_bscale: float = 0.5
    symmetry_axes: str = '12'

    def build_numerical_grid(self):
        par.set_parval_from_str('reference_metric::CoordSystem', self.coord_system)
        rfm.reference_metric()

    def build_symmetry(self):
        par.set_parval_from_str('indexedexp::symmetry_axes', self.symmetry_axes)

@dataclass
class Derivatives:
    rk_method: str = 'RK4'
    fd_order: int = 4
    real: str = 'double'
    cfl_factor: float = 0.5,
    lapse_condition: str = 'OnePlusLog',
    shift_condition: str = 'GammaDriving2ndOrder_Covariant',
    dirname: str = 'MoLtimestepping',
    ccodesdir: CodesDir = None

    def build_dir(self):
        self.root = os.path.join(self.ccodesdir.root,self.dirname)
        cmd.mkdir(self.root)

    def build_rhs_string(self):
        ricci_eval = 'Ricci_eval(&rfmstruct, &params, RK_INPUT_GFS, auxevol_gfs);'
        rhs_eval = 'rhs_eval(&rfmstruct, &params, auxevol_gfs, RK_INPUT_GFS, RK_OUTPUT_GFS);'
        self.rhs_string = '\n'.join(['\n', ricci_eval, rhs_eval, '\n'])

    def build_post_rhs_string(self):
        apply_bcs_curvilinear = 'apply_bcs_curvilinear(&params, &bcstruct, NUM_EVOL_GFS, evol_gf_parity, RK_OUTPUT_GFS);'
        enforce_detgammahat_constraint = 'enforce_detgammahat_constraint(&rfmstruct, &params,                     RK_OUTPUT_GFS);'
        self.post_rhs_string = '\n'.join(['\n', apply_bcs_curvilinear, enforce_detgammahat_constraint, '\n'])

    def build_c_code_generation(self):
        rk_order  = Butcher_dict[self.rk_method][1]
        self.build_dir()
        self.build_rhs_string()
        self.build_post_rhs_string()
        MoL.MoL_C_Code_Generation(self.rk_method,
                                  RHS_string = self.rhs_string,
                                  post_RHS_string = self.post_rhs_string,
                                  outdir = self.root
                                  )
    def build_finite_difference_order(self):
        par.set_parval_from_str('finite_difference::FD_CENTDERIVS_ORDER', self.fd_order)

@dataclass
class SimdIntrinsics:
    ccodesdir: CodesDir = None
    source_root: InitVar[str] = '../nrpytutorial/'
    source: os.PathLike = None
    destination: os.PathLike = None
    simd_path: os.PathLike = os.path.join('SIMD', 'SIMD_intrinsics', 'SIMD_intrinsics.h')


    def __post_init__(self, source_root):
        if self.source is None:
            self.source = os.path.join(source_root, 'SIMD', 'SIMD_intrinsics')

        if self.destination is None:
            self.destination = os.path.join(self.ccodesdir.root., 'SIMD', 'SIMD_intrinsics')
    
    



    def __init__(self, ccodesdir=None, source='../nrpytutorial/'):
        self.ccodesdir=ccodesdir
        self.source = os.path.join(source, 'SIMD', 'SIMD_intrinsics.h')
        self.target = os.path.join(self.ccodesdir.root, 'SIMD')

    def build(self):
        cmd.mkdir(self.target)
        shutil.copy(self.source, self.target)

class ScalarFiedInitialData:
    def __init__(self,
                 outputdir = None,
                 outputfilename = 'SFID.txt',
                 id_family = 'Gaussian_pulse',
                 pulse_amplitude = 0.4,
                 pulse_center = 0,
                 pulse_width = 1,
                 nr = 30000,
                 domain_size = None,
                 rmax_weight = 1.1
                 ):
        self.outputfilename = os.path.join(outputdir.outdir, outputfilename)
        self.id_family = id_family

# Step 2.b: Set the initial data parameters
# outputfilename  = os.path.join(,"SFID.txt")
ID_Family       = "Gaussian_pulse"
pulse_amplitude = 0.4
pulse_center    = 0
pulse_width     = 1
Nr              = 30000
# rmax            = domain_size*1.1

# Step 2.c: Generate the initial data
# sfid.ScalarField_InitialData(outputfilename,ID_Family,
#                              pulse_amplitude,pulse_center,pulse_width,Nr,rmax)

# Step 2.d: Generate the needed C code
# sfid.NRPy_param_funcs_register_C_functions_and_NRPy_basic_defines(Ccodesdir=Ccodesdir)



def build_scalar_field_collapse():
    codesdir = CodesDir()
    spatial = SpatialDimension(ccodesdir=codesdir)
    coord_system = CoordSystem()
    derivatives = Derivatives(ccodesdir=codesdir)
    simd = SimdIntrinsics(ccodesdir=codesdir)

    codesdir.build()
    spatial.build_grid()
    derivatives.build_c_code_generation()
    coord_system.build_numerical_grid()
    derivatives.build_finite_difference_order()
    simd.build()
    coord_system.build_symmetry()
    spatial.build_find_timestep()

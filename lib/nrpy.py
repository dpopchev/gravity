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

class SpatialDimension:
    def __init__(self, value=3, ccodesdir=None):
        self.value = value
        self.dim = None
        self.ccodesdir = ccodesdir

    def build_grid(self):
        par.set_parval_from_str('grid::DIM', self.value)
        self.dim = par.parval_from_str('grid::DIM')

    def build_find_timestep(self):
        target = os.path.join(self.ccodesdir.root, 'find_timestep.h')
        rfm.out_timestep_func_to_file(target)

class CoordSystem:
    def __init__(self,
                 coord_system = 'Spherical',
                 domain_size = 32,
                 sinh_width = 0.2,
                 sinhv2_const_dr = 0.2,
                 symtp_bscale = 0.5,
                 symmetry_axes = 12):
        self.coord_system = coord_system
        self.domain_size = domain_size
        self.sinh_width = sinh_width
        self.sinhv2_const_dr = sinhv2_const_dr
        self.symtp_bscale = symtp_bscale
        self.symmetry_axes = str(symmetry_axes)

    def build_numerical_grid(self):
        par.set_parval_from_str('reference_metric::CoordSystem', self.coord_system)
        rfm.reference_metric()

    def build_symmetry(self):
        par.set_parval_from_str('indexedexp::symmetry_axes', self.symmetry_axes)

class Derivatives:
    def __init__(self,
                 rk_method = 'RK4',
                 fd_order = 4,
                 real = 'double',
                 cfl_factor = 0.5,
                 lapse_condition = 'OnePlusLog',
                 shift_condition = 'GammaDriving2ndOrder_Covariant',
                 dirname = 'MoLtimestepping',
                 ccodesdir=None):
        self.rk_method = rk_method
        self.fd_order = fd_order
        self.real = real
        self.cfl_factor = cfl_factor
        self.lapse_condition = lapse_condition
        self.shift_condition = shift_condition
        self.dirname = dirname
        self.ccodesdir = ccodesdir

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

class SimdIntrinsics:
    def __init__(self, ccodesdir=None, source='../nrpytutorial/'):
        self.ccodesdir=ccodesdir
        self.source = os.path.join(source, 'SIMD', 'SIMD_intrinsics.h')
        self.target = os.path.join(self.ccodesdir.root, 'SIMD')

    def build(self):
        cmd.mkdir(self.target)
        shutil.copy(self.source, self.target)

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

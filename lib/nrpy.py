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

import pdb

@dataclass
class CcodesDir:
    root: str = "ccodesdir_default"
    output: str = None
    output_name: InitVar[str] = 'output'

    def __post_init__(self, output_name):
        if self.output is None:
            self.output = os.path.join(self.root, output_name)

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

    def build_paramter(self):
        par.set_parval_from_str('grid::DIM', self.value)

    def build_dim(self):
        self.dim = par.parval_from_str('grid::DIM')

    def build(self):
        self.build_paramter()
        self.build_dim()

@dataclass
class Derivatives:
    rk_method: str = 'RK4'
    fd_order: int = 4
    real: str = 'double'
    cfl_factor: float = 0.5
    lapse_condition: str = 'OnePlusLog'
    shift_condition: str = 'GammaDriving2ndOrder_Covariant'

@dataclass
class CoordSystem:
    name: str = 'Spherical'
    domain_size: int = 32
    sinh_width: float = 0.2
    sinhv2_const_dr: float = 0.05
    symtp_bscale: float = 0.5
    derivatives: Derivatives = None
    symmetry_axes: str = '12'

    def build_reference_metric(self):
        par.set_parval_from_str('reference_metric::CoordSystem', self.name)
        rfm.reference_metric()

    def build_fd_order(self):
        par.set_parval_from_str('finite_difference::FD_CENTDERIVS_ORDER',
                                self.derivatives.fd_order)

    def build_symmetry_axes(self):
        par.set_parval_from_str('indexedexp::symmetry_axes', self.symmetry_axes)

    def build(self):
        self.build_reference_metric()
        self.build_fd_order()
        self.build_symmetry_axes()

@dataclass
class RungeKuttaTimesteppingCode:
    derivatives: Derivatives = None
    ccodesdir: CcodesDir = None
    rk_order: int = None
    dirname: str = 'MoLtimestepping'
    dirpath: str = None
    rhs_string: str = None
    post_rhs_string: str = None

    def build_rk_order(self):
        self.rk_order = Butcher_dict[self.derivatives.rk_method]

    def build_dirpath(self):
        self.dirpath = os.path.join(self.ccodesdir.root, self.dirname)
        cmd.mkdir(self.dirpath)

    def build_rhs_string(self):
        ricci_eval = 'Ricci_eval(&rfmstruct, &params, RK_INPUT_GFS, auxevol_gfs);'
        rhs_eval = 'rhs_eval(&rfmstruct, &params, auxevol_gfs, RK_INPUT_GFS, RK_OUTPUT_GFS);'
        self.rhs_string = '\n'.join(['\n', ricci_eval, rhs_eval, '\n'])

    def build_post_rhs_string(self):
        apply_bcs_curvilinear = 'apply_bcs_curvilinear(&params, &bcstruct, NUM_EVOL_GFS, evol_gf_parity, RK_OUTPUT_GFS);'
        enforce_detgammahat_constraint = 'enforce_detgammahat_constraint(&rfmstruct, &params,                     RK_OUTPUT_GFS);'
        self.post_rhs_string = '\n'.join(['\n', apply_bcs_curvilinear, enforce_detgammahat_constraint, '\n'])

    def build_mol_c_code(self):
        self.build_rhs_string()
        self.build_post_rhs_string()
        MoL.MoL_C_Code_Generation(self.derivatives.rk_method,
                                  RHS_string = self.rhs_string,
                                  post_RHS_string = self.post_rhs_string,
                                  outdir = self.dirpath
                                  )

    def build_find_timestep_header(self):
        target = os.path.join(self.ccodesdir.root, 'find_timestep.h')
        rfm.out_timestep_func_to_file(target)

    def build(self):
        self.build_rk_order()
        self.build_dirpath()
        self.build_mol_c_code()
        self.build_find_timestep_header()

@dataclass
class ScalarFieldInitialData:
    ccodesdir: CcodesDir = None
    coord_system: CoordSystem = None
    outputname: InitVar[str] = 'SFID.txt'
    outputfilename: str = None
    id_family: str = "Gaussian_pulse"
    pulse_amplitude: float = 0.4
    pulse_center: float = 0
    pulse_width: float = 1
    nr: int = 30000
    rmax_weight: float = 1.1
    rmax: float = None

    def __post_init__(self, outputname):
        if self.outputfilename is None:
            self.outputfilename = os.path.join(self.ccodesdir.output, outputname)

    def build_rmax(self):
        self.rmax = self.coord_system.domain_size*self.rmax_weight

    def build(self):
        self.build_rmax()
        sfid.ScalarField_InitialData(self.outputfilename,
                                     self.id_family,
                                     self.pulse_amplitude,
                                     self.pulse_center,
                                     self.pulse_width,
                                     self.nr,
                                     self.rmax)

        sfid.NRPy_param_funcs_register_C_functions_and_NRPy_basic_defines(Ccodesdir=self.ccodesdir.root)

@dataclass
class Simd:
    ccodesdir: CcodesDir = None
    filename: str = 'SIMD_intrinsics.h'
    filepath: str = os.path.join('../nrpytutorial', 'SIMD')
    filetarget: str = None

    def build(self):
        self.target = os.path.join(self.ccodesdir.root, 'SIMD')
        cmd.mkdir(self.target)
        shutil.copy(os.path.join(self.filepath, self.filename), self.target)

def build_scalar_field_collapse():
    ccodesdir = CcodesDir()
    spatial_dimension = SpatialDimension()
    derivatives = Derivatives()
    coord_system = CoordSystem(derivatives=derivatives)
    moltimestepping = RungeKuttaTimesteppingCode(derivatives=derivatives,
                                                 ccodesdir=ccodesdir)
    simd = Simd(ccodesdir=ccodesdir)
    sfinitdata = ScalarFieldInitialData(ccodesdir=ccodesdir, coord_system=coord_system)

    ccodesdir.build()
    spatial_dimension.build()
    moltimestepping.build()
    coord_system.build()
    simd.build()
    sfinitdata.build()

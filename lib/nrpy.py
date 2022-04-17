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
    cfl_factor: float = 0.5
    lapse_condition: str = 'OnePlusLog'
    shift_condition: str = 'GammaDriving2ndOrder_Covariant'
    dirname: str = 'MoLtimestepping'
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
    simd_source: InitVar[str] = '../nrpytutorial'
    simd_path: InitVar[str] = os.path.join('SIMD')
    simd_header: InitVar[str] = 'SIMD_intrinsics.h'
    source: str = None
    destination: str = None

    def __post_init__(self, simd_source, simd_path, simd_header):
        if self.source is None:
            self.source = os.path.join(simd_source, simd_path, simd_header)

        if self.destination is None:
            self.destination = os.path.join(self.ccodesdir.root, simd_path)

    def build_destination(self):
        cmd.mkdir(self.destination)

    def build(self):
        self.build_destination()
        shutil.copy(self.source, self.destination)

@dataclass
class ScalarFieldInitialData:
    ccodesdir: CodesDir = None
    coord_system: InitVar[CoordSystem] = None
    outfilename: InitVar[str] = 'SFID.txt'
    output: str = None
    id_family: str = "Gaussian_pulse"
    pulse_amplitude: float = 0.4
    pulse_center: float = 0
    pulse_width: float = 1
    nr: int = 30000
    rmax_weight: InitVar[float] = 1.1
    rmax: float = None

    def __post_init__(self, coord_system, outfilename, rmax_weight ):
        if self.output is None:
            self.output = os.path.join( self.ccodesdir.output, outfilename )

        if self.rmax is None:
            self.rmax = coord_system.domain_size*rmax_weight

    def build_initial_data(self):
        sfid.ScalarField_InitialData(self.output,
                                     self.id_family,
                                     self.pulse_amplitude,
                                     self.pulse_center,
                                     self.pulse_width,
                                     self.nr,
                                     self.rmax)

    def build_c_code(self):
        sfid.NRPy_param_funcs_register_C_functions_and_NRPy_basic_defines(Ccodesdir=self.ccodesdir.root)

    def build(self):
        self.build_initial_data()
        self.build_c_code()


def build_scalar_field_collapse():
    codesdir = CodesDir()
    spatial = SpatialDimension(ccodesdir=codesdir)
    coord_system = CoordSystem()
    derivatives = Derivatives(ccodesdir=codesdir)
    simd = SimdIntrinsics(ccodesdir=codesdir)
    sf_init_data = ScalarFieldInitialData(ccodesdir=codesdir,
                                          coord_system=coord_system)

    codesdir.build()
    spatial.build_grid()
    derivatives.build_c_code_generation()
    coord_system.build_numerical_grid()
    derivatives.build_finite_difference_order()
    simd.build()
    coord_system.build_symmetry()
    spatial.build_find_timestep()
    sf_init_data.build()

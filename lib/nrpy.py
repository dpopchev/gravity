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

CCODESDIR = os.path.join("ccodesdir_default")

class CodesDir:
    def __init__(self, root=CCODESDIR, output = 'output'):
        self.root = root
        self.outdir = os.path.join(self.root, output)

    def clean(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def build(self):
        self.clean()
        for directory in (self.root, self.outdir):
            cmd.mkdir(directory)

class SpatialDimension:
    def __init__(self, value=3):
        self.value = value
        self.dim = None
    def build(self):
        par.set_parval_from_str('grid::DIM', self.value)
        self.dim = par.parval_from_str('grid::DIM')

class CoordSystem:
    def __init__(self,
                 coord_system = 'Spherical',
                 domain_size = 32,
                 sinh_width = 0.2,
                 sinhv2_const_dr = 0.2,
                 symtp_bscale = 0.5):
        self.coord_system = coord_system
        self.domain_size = domain_size
        self.sinh_width = sinh_width
        self.sinhv2_const_dr = sinhv2_const_dr
        self.symtp_bscale = symtp_bscale

    def build_numerical_grid(self):
        par.set_parval_from_str('reference_metric::CoordSystem', self.coord_system)
        rfm.reference_metric()

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

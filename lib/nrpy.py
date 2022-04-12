import ScalarField.ScalarField_InitialData as sfid
import reference_metric as rfm
import shutil
import cmdline_helper as cmd
import NRPy_param_funcs as par
import MoLtimestepping.C_Code_Generation as MoL
from MoLtimestepping.RK_Butcher_Table_Dictionary import Butcher_dict

import os

CCODESDIR = os.path.join("ccodesdir_default")

class ScalarFieldInitialData:

    def __init__(
        self,
        out_dir=None,
        out_file='SFID.txt',
        id_family='Gaussian_pulse',
        pulse_amplitude=0.4,
        pulse_center=0,
        pulse_width=1,
        nr=30000,
        domain_size=32,
        domain_weight=1.1,
        ccodesdir=CCODESDIR
            ):

        self.outputfilename = os.path.join(out_dir, out_file)
        self.id_family = id_family
        self.pulse_amplitude = pulse_amplitude
        self.pulse_center = pulse_center
        self.pulse_width = pulse_width
        self.nr = nr
        self.rmax = domain_size * domain_weight
        self.ccodesdir = ccodesdir

    def build(self):
        self.build_initial_data()
        self.build_c_code()

    def build_initial_data(self):
        sfid.ScalarField_InitialData(self.outputfilename,
                                     self.id_family,
                                     self.pulse_amplitude,
                                     self.pulse_center,
                                     self.pulse_width
                                     self.nr,
                                     self.rmax)

    def build_c_code(self):
        sfid.NRPy_param_funcs_register_C_functions_and_NRPy_basic_defines(
            Ccodesdir=self.ccodesdir)

class FindTimestepHeader:
    def __init__(self, out_dir=CCODESDIR):
        self.ccodesdir = os.path.join(self.out_dir)
    def build(self):
        rfm.out_timestep_func_to_file(os.path.join(self.ccodesdir,"find_timestep.h"))

class NumericalGrids:

    def __init__(self,
                 out_dir=CCODESDIR,
                 output_dir = 'output',
                 spatial_dimensions = ('grid::DIM', 3),
                 coord_system = 'Spherical',
                 domain_size = 32,
                 sinh_width = 0.2,
                 sinhv2_const_dr = 0.05,
                 sym_tp_bscale = 0.5,
                 rk_method = 'RK4',
                 fd_order = 4,
                 real = 'double',
                 cfl_factor = 0.5,
                 lapse_condition = 'OnePlusLog'
                 shift_condition = 'GammaDriving2ndOrder_Covariant'
                 ):
        self.ccodesdir = os.path.join(out_dir)
        self.output = os.path.join(self.ccodesdir, output_dir)
        self.spatial_dimensions = spatial_dimensions
        self.coord_system = coord_system
        self.domain_size = domain_size
        self.sinh_width = sinh_width
        self.sinhv2_const_dr = sinhv2_const_dr
        self.sym_tp_bscale = sym_tp_bscale
        self.rk_method = rk_method
        self.fd_order = fk_order
        self.real = real
        self.cfl_factor = cfl_factor
        self.lapse_condition = lapse_condition
        self.shift_condition = shift_condition

    def build(self):
        self.build_ccodesdir()
        self.build_output()
        self.build_dim()
        self.build_rk_timestepping_code()
        self.build_coord_system_numerical_grid()
        par.set_parval_from_str('finite_difference::FD_CENTDRIVS_ORDER', self.fd_order)
        self.build_simd()

    
    def build_ccodesdir(self):
        shutil.rmtree(self.ccodesdir, ignore_errors=True)
        cmd.mkdir(self.ccodesdir)

    def build_output(self):
        cmd.mkdir(outdir)

    def build_dim(self):
        par.set_parval_from_str(*self.spatial_dimensions)
        self.dim = par.parval_from_str(self.spatial_dimensions[0])

    def build_rk_timestepping_code(self):
        rk_order = Butcher_dict[self.rk_method][1]
        moltimestepping = 'MoLtimestepping/'
        cmd.mkdir(os.path.join(self.ccodesdir, moltimestepping))
        RHS_string = """
Ricci_eval(&rfmstruct, &params, RK_INPUT_GFS, auxevol_gfs);
rhs_eval(&rfmstruct, &params, auxevol_gfs, RK_INPUT_GFS, RK_OUTPUT_GFS);"""
        post_RHS_string = """
apply_bcs_curvilinear(&params, &bcstruct, NUM_EVOL_GFS, evol_gf_parity, RK_OUTPUT_GFS);
enforce_detgammahat_constraint(&rfmstruct, &params,                     RK_OUTPUT_GFS);
"""
        MoL.MoL_C_Code_Generation(
            self.rk_method,
            RHS_string = RHS_string,
            post_RHS_string = post_RHS_string,
            outdir = os.path.join(self.ccodesdir, moltimestepping)

            )

    def build_coord_system_numerical_grid(self):
        par.set_parval_from_str('reference_metric::CoordSystem', self.coord_system)
        rfm.reference_metric()

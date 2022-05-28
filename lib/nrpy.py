# Step P1: Import needed NRPy+ core modules:
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
import BSSN.ADM_Numerical_Spherical_or_Cartesian_to_BSSNCurvilinear as AtoBnum
import time
import BSSN.BSSN_RHSs as rhs
import BSSN.BSSN_gauge_RHSs as gaugerhs
import BSSN.BSSN_quantities as Bq
import ScalarField.ScalarField_RHSs as sfrhs
import ScalarField.ScalarField_Tmunu as sfTmunu
import BSSN.BSSN_stress_energy_source_terms as Bsest
import BSSN.Enforce_Detgammahat_Constraint as EGC
import BSSN.BSSN_constraints as bssncon
import CurviBoundaryConditions.CurviBoundaryConditions as cbcs
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import matplotlib.image as mgimg

import glob
import sys
from matplotlib import animation

from dataclasses import dataclass, InitVar, field
from typing import Any, List
from itertools import product

# BUILD_STEPS
# CcodesDir.build()
#

@dataclass
class CcodesDir:
    root: str = None
    outdir: str = None

    def __post_init__(self, _root, _outdir):
        if self.root is None:
            self.root = os.path.join(_root)

        if self.outdir is None:
            self.outdir = os.path.join(self.root, _outdir)

    def clean(self):
        shutil.rmtree(self.root, ignore_errors=True)

    def make_root(self):
        cmd.mkdir(self.root)

    def make_outdir(self):
        cmd.mkdir(self.outdir)

    @classmethod
    def build(cls, root = "ccodesdir_default", outdir = 'output'):
        _root, _outdir = map(os.path.join, (root, outdir))
        self = cls(_root, _outdir)
        self.make_root()
        self.make_outdir()
        return self


@dataclass
class SpatialDimension:
    parameter: str = None
    value: int = None

    @classmethod
    def build(cls, parameter = 'grid:DIM', value = 3):
        self = cls(parameter, value)
        return self

    @property
    def dim(self):
        par.set_parval_from_str(self.parameter, self.value)
        _dim = par.parval_from_str(self.parameter)
        return _dim

@dataclass
class CoordSystem:
    _name: InitVar[str] = "Spherical"
    name: str = None
    choices: List = field(default_factory=lambda: ['Spherical', 'SinhSpherical', 'SinhSphericalv2', 'Cylindrical', 'SinhCylindrical', 'SymTP', 'SinhSymTP'])
    domain_size: int = 32
    sinh_width: float = 0.2 # If Sinh* coordinates chosen
    sinhv2_const_dr: float = 0.05# If Sinh*v2 coordinates chosen
    symtp_bscale: float    = 0.5 # If SymTP chosen
    is_spherical_symmetry: bool = True

    def __post_init__(self, _name):
        if self.name is None:
            self.name = _name

        if self.name not in self.choices:
            raise AttributeError(f'{self.name} should be one of {self.choices}')

    def build(self):
        par.set_parval_from_str("reference_metric::CoordSystem",self.name)
        rfm.reference_metric()

        if self.is_spherical_symmetry:
            par.set_parval_from_str("indexedexp::symmetry_axes","12")

@dataclass
class NumericalIntegration:
    _method: InitVar[str] = 'RK4'
    method: str = None
    fd_order: int = 4
    real: str = 'double'
    cfl_factor: float = 0.5
    choices: List = field(default_factory=lambda: ['Euler', 'RK2 Heun', 'RK2 MP', 'RK2 Ralston', 'RK3', 'RK3 Heun', 'RK3 Ralston', 'SSPRK3', 'RK4', 'DP5', 'DP5alt', 'CK5', 'DP6', 'L6', 'DP8'])
    lapse: str = "OnePlusLog"
    shift: str = "GammaDriving2ndOrder_Covariant"
    rk_order: Any = None
    ccodesdir: CcodesDir = None
    rhs_string: str = None
    post_rhs_string: str = None

    def __post_init__(self, _method):
        if self.method is None:
            self.method = _method

        if self.method not in self.choices:
            raise AttributeError(f'{self.name} should be one of {self.choices}')

    def build_rhs_string(self):
        ricci_eval = 'Ricci_eval(&rfmstruct, &params, RK_INPUT_GFS, auxevol_gfs);'
        rhs_eval = 'rhs_eval(&rfmstruct, &params, auxevol_gfs, RK_INPUT_GFS, RK_OUTPUT_GFS);'
        self.rhs_string = '\n'.join(['\n', ricci_eval, rhs_eval, '\n'])

    def build_post_rhs_string(self):
        apply_bcs_curvilinear = 'apply_bcs_curvilinear(&params, &bcstruct, NUM_EVOL_GFS, evol_gf_parity, RK_OUTPUT_GFS);'
        enforce_detgammahat_constraint = 'enforce_detgammahat_constraint(&rfmstruct, &params,                     RK_OUTPUT_GFS);'
        self.post_rhs_string = '\n'.join(['\n', apply_bcs_curvilinear, enforce_detgammahat_constraint, '\n'])

    def build_moltimestepping(self):
        dirname = 'MoLtimestepping'
        dirtarget = os.path.join(self.ccodesdir.root, dirname)
        cmd.mkdir(dirtarget)

        self.build_rhs_string()
        self.build_post_rhs_string()

        MoL.MoL_C_Code_Generation(self.method,
                                  RHS_string=self.rhs_string,
                                  post_RHS_string=self.post_rhs_string,
                                  outdir=dirtarget
                                  )
    def build_timestep(self):
        destination = os.path.join(self.ccodesdir.root, 'find_timestep.h')
        rfm.out_timestep_func_to_file(destination)

    def build(self):
        self.rk_order = Butcher_dict[self.method][1]
        self.build_moltimestepping()
        self.build_timestep()
        par.set_parval_from_str("finite_difference::FD_CENTDERIVS_ORDER", self.fd_order)

@dataclass
class Simd:
    source: str = '../nrpytutorial/SIMD/SIMD_intrinsics.h'
    ccodesdir: CcodesDir = None

    def build(self):
        destination = os.path.join(self.ccodesdir.root, 'SIMD')
        cmd.mkdir(destination)
        shutil.copy(self.source,destination)

@dataclass
class ScalarFieldInitData:
    _outputfilename: InitVar[str] = 'SFID.txt'
    outputfilename: str = None
    ccodesdir: CcodesDir = None
    id_family: str       = "Gaussian_pulse"
    pulse_amplitude: float = 0.4
    pulse_center: float    = 0
    pulse_width: float     = 1
    nr: int              = 30000
    coord_system: CoordSystem = None
    rmax_weight: float = 1.1
    rmax: float = None

    def __post_init__(self, _outputfilename):
        if self.outputfilename is None:
            self.outputfilename = os.path.join(self.ccodesdir.outdir, _outputfilename)

    def build(self):
        self.rmax = self.coord_system.domain_size*self.rmax_weight
        sfid.ScalarField_InitialData(self.outputfilename,
                                     self.id_family,
                                     self.pulse_amplitude,
                                     self.pulse_center,
                                     self.pulse_width,
                                     self.nr,
                                     self.rmax)
        sfid.NRPy_param_funcs_register_C_functions_and_NRPy_basic_defines(Ccodesdir=self.ccodesdir.root)


@dataclass
class AdmBssnCoordConverter:
    coord_system: CoordSystem = None
    ccodesdir: CcodesDir = None
    adm_quantities: str = 'ID_scalarfield_ADM_quantities'

    def build(self):
        AtoBnum.Convert_Spherical_or_Cartesian_ADM_to_BSSN_curvilinear(self.coord_system.name,
                                                                       self.adm_quantities,
                                                                       Ccodesdir=self.ccodesdir.root,
                                                                       loopopts="")

@dataclass
class BssnSpaceTime:
    numerical: NumericalIntegration = None
    ccodesdir: CcodesDir = None
    spatial_dimension: SpatialDimension = None
    t4uu: Any = None
    betau: Any = None
    enforce_detg_constraint_symb_expressions: Any = None
    hamiltonian: Any = None
    coord_system: CoordSystem = None

    def build_bssn_gauge_rhs(self):
        par.set_parval_from_str("BSSN.BSSN_gauge_RHSs::LapseEvolutionOption", self.numerical.lapse)
        par.set_parval_from_str("BSSN.BSSN_gauge_RHSs::ShiftEvolutionOption", self.numerical.shift)

    def build_rfm(self):
        rfmdir = os.path.join(self.ccodesdir.root,"rfm_files/")
        cmd.mkdir(rfmdir)
        par.set_parval_from_str("reference_metric::enable_rfm_precompute","True")
        par.set_parval_from_str("reference_metric::rfm_precompute_Ccode_outdir",rfmdir)

    def build_bssn(self):
        par.set_parval_from_str("BSSN.BSSN_quantities::LeaveRicciSymbolic","True")
        rhs.BSSN_RHSs()
        sfrhs.ScalarField_RHSs()

    def build_scalar_field_tmunu(self):
        sfTmunu.ScalarField_Tmunu()
        self.t4uu = sfTmunu.T4UU

    def build_gauge(self):
        Bsest.BSSN_source_terms_for_BSSN_RHSs(self.t4uu)
        rhs.trK_rhs += Bsest.sourceterm_trK_rhs

        dim_range = range(self.spatial_dimension.dim)
        for i in dim_range:
            rhs.Lambdabar_rhsU[i] += Bsest.sourceterm_Lambdabar_rhsU[i]
            rhs.lambda_rhsU[i]    += Bsest.sourceterm_lambda_rhsU[i]

        ij_ranges = product(dim_range, dim_range)
        for i, j in ij_ranges:
            rhs.a_rhsDD[i][j] += Bsest.sourceterm_a_rhsDD[i][j]

        gaugerhs.BSSN_gauge_RHSs()
        Bq.BSSN_basic_tensors()
        self.betau = Bq.betaU

    def build_ricci(self):
        par.set_parval_from_str("BSSN.BSSN_quantities::LeaveRicciSymbolic","False")
        Bq.RicciBar__gammabarDD_dHatD__DGammaUDD__DGammaU()

    def build_hamiltonina_gridfunc(self):
        self.hamiltonian = gri.register_gridfunctions("AUX","H")
        bssncon.BSSN_constraints(add_T4UUmunu_source_terms=False)
        Bsest.BSSN_source_terms_for_BSSN_constraints(self.t4uu)
        bssncon.H += Bsest.sourceterm_H

    def build_kreis_oliger_dissipation(self):
        diss_strength = par.Cparameters("REAL","ScalarFieldCollapse",["diss_strength"],0.1)

        alpha_dKOD   = ixp.declarerank1("alpha_dKOD")
        cf_dKOD      = ixp.declarerank1("cf_dKOD")
        trK_dKOD     = ixp.declarerank1("trK_dKOD")
        sf_dKOD      = ixp.declarerank1("sf_dKOD")
        sfM_dKOD     = ixp.declarerank1("sfM_dKOD")
        betU_dKOD    = ixp.declarerank2("betU_dKOD","nosym")
        vetU_dKOD    = ixp.declarerank2("vetU_dKOD","nosym")
        lambdaU_dKOD = ixp.declarerank2("lambdaU_dKOD","nosym")
        aDD_dKOD     = ixp.declarerank3("aDD_dKOD","sym01")
        hDD_dKOD     = ixp.declarerank3("hDD_dKOD","sym01")

        range_3 = range(3)
        for k in range_3:
            gaugerhs.alpha_rhs += diss_strength*alpha_dKOD[k]*rfm.ReU[k]
            rhs.cf_rhs         += diss_strength*   cf_dKOD[k]*rfm.ReU[k]
            rhs.trK_rhs        += diss_strength*  trK_dKOD[k]*rfm.ReU[k]
            sfrhs.sf_rhs       += diss_strength*   sf_dKOD[k]*rfm.ReU[k]
            sfrhs.sfM_rhs      += diss_strength*  sfM_dKOD[k]*rfm.ReU[k]

        for k, i in product(range_3, range_3):
            if "2ndOrder" in self.numerical.shift:
                gaugerhs.bet_rhsU[i] += diss_strength*   betU_dKOD[i][k]*rfm.ReU[k]
            gaugerhs.vet_rhsU[i]     += diss_strength*   vetU_dKOD[i][k]*rfm.ReU[k]
            rhs.lambda_rhsU[i]       += diss_strength*lambdaU_dKOD[i][k]*rfm.ReU[k]

        for k, i, j in product(range_3, range_3, range_3):
            rhs.a_rhsDD[i][j] += diss_strength*aDD_dKOD[i][j][k]*rfm.ReU[k]
            rhs.h_rhsDD[i][j] += diss_strength*hDD_dKOD[i][j][k]*rfm.ReU[k]

    def build_rfm_quantities(self):
        par.set_parval_from_str("reference_metric::enable_rfm_precompute","False") # Reset to False to disable rfm_precompute.
        rfm.ref_metric__hatted_quantities()

    def build_c_code_bssn_plus_scalarfield_rhs(self):
        print("Generating C code for BSSN RHSs in "
              + par.parval_from_str("reference_metric::CoordSystem")
              +" coordinates.")

        start = time.time()

        # Construct the left-hand sides and right-hand-side expressions for all BSSN RHSs
        lhs_names = [        "alpha",       "cf",       "trK",         "sf",         "sfM"   ]
        rhs_exprs = [gaugerhs.alpha_rhs, rhs.cf_rhs, rhs.trK_rhs, sfrhs.sf_rhs, sfrhs.sfM_rhs]

        for i in range(3):
            lhs_names.append(        "betU"+str(i))
            rhs_exprs.append(gaugerhs.bet_rhsU[i])
            lhs_names.append(   "lambdaU"+str(i))
            rhs_exprs.append(rhs.lambda_rhsU[i])
            lhs_names.append(        "vetU"+str(i))
            rhs_exprs.append(gaugerhs.vet_rhsU[i])
            for j in range(i,3):
                lhs_names.append(   "aDD"+str(i)+str(j))
                rhs_exprs.append(rhs.a_rhsDD[i][j])
                lhs_names.append(   "hDD"+str(i)+str(j))
                rhs_exprs.append(rhs.h_rhsDD[i][j])

        # Sort the lhss list alphabetically, and rhss to match.
        #   This ensures the RHSs are evaluated in the same order
        #   they're allocated in memory:
        lhs_names,rhs_exprs = [list(x) for x in zip(*sorted(zip(lhs_names,rhs_exprs), key=lambda pair: pair[0]))]

        # Declare the list of lhrh's
        BSSN_evol_rhss = []
        for var in range(len(lhs_names)):
            BSSN_evol_rhss.append(lhrh(lhs=gri.gfaccess("rhs_gfs",lhs_names[var]),rhs=rhs_exprs[var]))

        # Set up the C function for the BSSN RHSs
        desc="Evaluate the BSSN RHSs"
        name="rhs_eval"
        out_c_target = os.path.join(self.ccodesdir.root, name + '.h')
        outCfunction(
            outfile  = out_c_target, desc=desc, name=name,
            params   = """rfm_struct *restrict rfmstruct,const paramstruct *restrict params,
                          const REAL *restrict auxevol_gfs,const REAL *restrict in_gfs,REAL *restrict rhs_gfs""",
            body     = fin.FD_outputC("returnstring",BSSN_evol_rhss, params="outCverbose=False,enable_SIMD=True",
                                      upwindcontrolvec=self.betau),
            loopopts = "InteriorPoints,enable_SIMD,enable_rfm_precompute")
        end = time.time()
        print("(BENCH) Finished BSSN_RHS C codegen in " + str(end - start) + " seconds.")

    def build_c_code_ricci(self):
        print("Generating C code for Ricci tensor in "
              +par.parval_from_str("reference_metric::CoordSystem")
              +" coordinates.")

        start = time.time()
        desc="Evaluate the Ricci tensor"
        name="Ricci_eval"
        out_c_target = os.path.join(self.ccodesdir.root, name + '.h')
        outCfunction(
            outfile  = out_c_target, desc=desc, name=name,
            params   = """rfm_struct *restrict rfmstruct,const paramstruct *restrict params,
                          const REAL *restrict in_gfs,REAL *restrict auxevol_gfs""",
            body     = fin.FD_outputC("returnstring",
                                      [lhrh(lhs=gri.gfaccess("auxevol_gfs","RbarDD00"),rhs=Bq.RbarDD[0][0]),
                                       lhrh(lhs=gri.gfaccess("auxevol_gfs","RbarDD01"),rhs=Bq.RbarDD[0][1]),
                                       lhrh(lhs=gri.gfaccess("auxevol_gfs","RbarDD02"),rhs=Bq.RbarDD[0][2]),
                                       lhrh(lhs=gri.gfaccess("auxevol_gfs","RbarDD11"),rhs=Bq.RbarDD[1][1]),
                                       lhrh(lhs=gri.gfaccess("auxevol_gfs","RbarDD12"),rhs=Bq.RbarDD[1][2]),
                                       lhrh(lhs=gri.gfaccess("auxevol_gfs","RbarDD22"),rhs=Bq.RbarDD[2][2])],
                                       params="outCverbose=False,enable_SIMD=True"),
            loopopts = "InteriorPoints,enable_SIMD,enable_rfm_precompute")
        end = time.time()
        print("(BENCH) Finished Ricci C codegen in " + str(end - start) + " seconds.")

    def build_c_code_hamiltonian(self):
        start = time.time()
        print("Generating optimized C code for Hamiltonian constraint. May take a while, depending on CoordSystem.")
        # Set up the C function for the Hamiltonian RHS
        desc="Evaluate the Hamiltonian constraint"
        name="Hamiltonian_constraint"
        outCfunction(
            outfile  = os.path.join(self.ccodesdir.root,name+".h"), desc=desc, name=name,
            params   = """rfm_struct *restrict rfmstruct,const paramstruct *restrict params,
                          REAL *restrict in_gfs, REAL *restrict auxevol_gfs, REAL *restrict aux_gfs""",
            body     = fin.FD_outputC("returnstring",lhrh(lhs=gri.gfaccess("aux_gfs", "H"), rhs=bssncon.H),
                                      params="outCverbose=False"),
            loopopts = "InteriorPoints,enable_rfm_precompute")

        end = time.time()
        print("(BENCH) Finished Hamiltonian C codegen in " + str(end - start) + " seconds.")

    def build_c_code_gammadet(self):
        start = time.time()
        print("Generating optimized C code for gamma constraint. May take a while, depending on CoordSystem.")

        # Set up the C function for the det(gammahat) = det(gammabar)
        EGC.output_Enforce_Detgammahat_Constraint_Ccode(self.ccodesdir.root,exprs=self.enforce_detg_constraint_symb_expressions)
        end = time.time()
        print("(BENCH) Finished gamma constraint C codegen in " + str(end - start) + " seconds.")

    def build_c_code_parameters(self):
        # Step 4.e.i: Generate declare_Cparameters_struct.h, set_Cparameters_default.h, and set_Cparameters[-SIMD].h
        par.generate_Cparameters_Ccodes(os.path.join(self.ccodesdir.root))

        rfm.out_default_free_parameters_for_rfm(os.path.join(self.ccodesdir.root,"free_parameters.h"),
                                                self.coord_system.domain_size,
                                                self.coord_system.sinh_width,
                                                self.coord_system.sinhv2_const_dr,
                                                self.coord_system.symtp_bscale
                                                )

        rfm.set_Nxx_dxx_invdx_params__and__xx_h(self.ccodesdir.root)

        rfm.xx_to_Cart_h("xx_to_Cart","./set_Cparameters.h",os.path.join(self.ccodesdir.root,"xx_to_Cart.h"))

        par.generate_Cparameters_Ccodes(os.path.join(self.ccodesdir.root))

    def build(self):
        self.build_bssn_gauge_rhs()

        print("Generating symbolic expressions for BSSN RHSs...")
        start = time.time()

        self.build_rfm()
        self.build_bssn()
        self.build_scalar_field_tmunu()
        self.build_gauge()

        self.enforce_detg_constraint_symb_expressions = EGC.Enforce_Detgammahat_Constraint_symb_expressions()

        self.build_ricci()
        self.build_hamiltonina_gridfunc()
        self.build_kreis_oliger_dissipation()
        self.build_rfm_quantities()

        end = time.time()
        print("(BENCH) Finished BSSN symbolic expressions in "+str(end-start)+" seconds.")

        self.build_c_code_bssn_plus_scalarfield_rhs()
        self.build_c_code_ricci()
        self.build_c_code_hamiltonian()
        self.build_c_code_gammadet()
        self.build_c_code_parameters()

@dataclass
class BoundaryCondition:
    ccodesdir: CcodesDir = None

    def build(self):
        cbcs.Set_up_CurviBoundaryConditions(os.path.join(self.ccodesdir.root,"boundary_conditions/"),Cparamspath=os.path.join("../"),path_prefix='../nrpytutorial')


@dataclass
class MainCcode:
    ccodesdir: CcodesDir = None
    numerical: NumericalIntegration = None
    file_list: List = field(default_factory=list)

    def build_main(self):

        with open(os.path.join(self.ccodesdir.root,"ScalarFieldCollapse_Playground_REAL__NGHOSTS__CFL_FACTOR.h"), "w") as file:
            file.write("""
        // Part P0.a: Set the number of ghost cells, from NRPy+'s FD_CENTDERIVS_ORDER
        #define NGHOSTS """+str(int(self.numerical.fd_order/2)+1)+"""
        // Part P0.b: Set the numerical precision (REAL) to double, ensuring all floating point
        //            numbers are stored to at least ~16 significant digits
        #define REAL """+self.numerical.real+"""
        // Part P0.c: Set the number of ghost cells, from NRPy+'s FD_CENTDERIVS_ORDER
        REAL CFL_FACTOR = """+str(self.numerical.cfl_factor)+"""; // Set the CFL Factor. Can be overwritten at command line.\n""")

        shutil.copy('templates/ScalarFieldCollapse_Playground.c', f'{self.ccodesdir.root}/ScalarFieldCollapse_Playground.c')

    def compile(self):
        print("Now compiling, should take ~10 seconds...\n")
        start = time.time()
        cmd.C_compile(os.path.join(self.ccodesdir.root,"ScalarFieldCollapse_Playground.c"),
                      os.path.join(self.ccodesdir.outdir,"ScalarFieldCollapse_Playground"),compile_mode="optimized")
        end = time.time()
        print("(BENCH) Finished in "+str(end-start)+" seconds.\n")

        # Change to output directory
        os.chdir(self.ccodesdir.outdir)
        # Clean up existing output files
        cmd.delete_existing_files("out*.txt")
        cmd.delete_existing_files("out*.png")
        # Run executable

        print(os.getcwd())
        print("Now running, should take ~20 seconds...\n")
        start = time.time()
        cmd.Execute("ScalarFieldCollapse_Playground", "640 2 2 "+str(self.numerical.cfl_factor),"out640.txt")
        end = time.time()
        print("(BENCH) Finished in "+str(end-start)+" seconds.\n")

        # Return to root directory
        os.chdir(os.path.join("../../"))

    def build_animation(self):

        globby = glob.glob(os.path.join(self.ccodesdir.outdir,'out640-00*.txt'))
        self.file_list = []
        for x in sorted(globby):
            self.file_list.append(x)

        for filename in self.file_list:
            fig = plt.figure(figsize=(8,6))
            x,r,sf,sfM,alpha,cf,logH = np.loadtxt(filename).T #Transposed for easier unpacking

            ax  = fig.add_subplot(321)
            ax2 = fig.add_subplot(322)
            ax3 = fig.add_subplot(323)
            ax4 = fig.add_subplot(324)
            ax5 = fig.add_subplot(325)

            ax.set_title("Scalar field")
            ax.set_ylabel(r"$\varphi(t,r)$")
            ax.set_xlim(0,20)
            ax.set_ylim(-0.6,0.6)
            ax.plot(r,sf,'k-')
            ax.grid()

            ax2.set_title("Scalar field conjugate momentum")
            ax2.set_ylabel(r"$\Pi(t,r)$")
            ax2.set_xlim(0,20)
            ax2.set_ylim(-1,1)
            ax2.plot(r,sfM,'b-')
            ax2.grid()

            ax3.set_title("Lapse function")
            ax3.set_ylabel(r"$\alpha(t,r)$")
            ax3.set_xlim(0,20)
            ax3.set_ylim(0,1.02)
            ax3.plot(r,alpha,'r-')
            ax3.grid()

            ax4.set_title("Conformal factor")
            ax4.set_xlabel(r"$r$")
            ax4.set_ylabel(r"$W(t,r)$")
            ax4.set_xlim(0,20)
            ax4.set_ylim(0,1.02)
            ax4.plot(r,cf,'g-',label=("$p = 0.043149493$"))
            ax4.grid()

            ax5.set_title("Hamiltonian constraint violation")
            ax5.set_xlabel(r"$r$")
            ax5.set_ylabel(r"$\mathcal{H}(t,r)$")
            ax5.set_xlim(0,20)
            ax5.set_ylim(-16,0)
            ax5.plot(r,logH,'m-')
            ax5.grid()

            plt.tight_layout()
            savefig(filename+".png",dpi=150)
            plt.close(fig)
            sys.stdout.write("%c[2K" % 27)
            sys.stdout.write("Processing file "+filename+"\r")
            sys.stdout.flush()

    def build_convergence(self):
        fig = plt.figure(frameon=False)
        ax = fig.add_axes([0, 0, 1, 1])
        ax.axis('off')

        myimages = []

        for i in range(len(self.file_list)):
            img = mgimg.imread(self.file_list[i]+".png")
            imgplot = plt.imshow(img)
            myimages.append([imgplot])

        ani = animation.ArtistAnimation(fig, myimages, interval=100,  repeat_delay=1000)
        plt.close()

        try:
            ani.save(os.path.join(self.ccodesdir.outdir,'ScalarField_Collapse.mp4'), fps=5, dpi=150)
        except ValueError:
            print('exception when saving animation, skipping')

        os.chdir(self.ccodesdir.outdir)

        cmd.delete_existing_files("out320*.txt")
        cmd.Execute("ScalarFieldCollapse_Playground", "320 2 2 "+str(self.numerical.cfl_factor),"out320.txt")

        os.chdir(os.path.join("..",".."))

        outfig = os.path.join(self.ccodesdir.outdir,"ScalarFieldCollapse_H_convergence.png")

        fig = plt.figure()

        r_640,H_640 = np.loadtxt(os.path.join(self.ccodesdir.outdir,"out640.txt")).T
        r_320,H_320 = np.loadtxt(os.path.join(self.ccodesdir.outdir,"out320.txt")).T

        plt.title("Plot demonstrating 4th order\nconvergence of constraint violations")
        plt.xlabel(r"$r$")
        plt.ylabel(r"$\log_{10}|\mathcal{H}|$")
        plt.xlim(0,16)
        plt.plot(r_640,H_640,label=r"$N_{r} = 640$")
        plt.plot(r_320,H_320+4*np.log10(320.0/640.0),label=r"$N_{r} = 320$, mult by $(320/640)^{4}$")
        plt.legend()

        plt.tight_layout()
        plt.savefig(outfig,dpi=150)
        plt.close(fig)

    def build(self):
        self.build_main()
        self.compile()
        self.build_animation()
        self.build_convergence()

def build_scalar_field_collapse():
    ccodesdir = CcodesDir()
    spatial_dimension = SpatialDimension()
    coord_system = CoordSystem()
    numerical = NumericalIntegration(ccodesdir=ccodesdir)
    simd = Simd(ccodesdir=ccodesdir)
    scalar_field_init_data = ScalarFieldInitData(ccodesdir=ccodesdir,
                                                 coord_system=coord_system)
    adm_bssn_converter = AdmBssnCoordConverter(coord_system=coord_system,
                                               ccodesdir=ccodesdir)
    bssn_spacetime = BssnSpaceTime(numerical=numerical, ccodesdir=ccodesdir,
                                   spatial_dimension=spatial_dimension,
                                   coord_system=coord_system)
    boundary_condition = BoundaryCondition(ccodesdir=ccodesdir)
    main_ccode = MainCcode(ccodesdir=ccodesdir, numerical=numerical)

    steps = (ccodesdir, )
    steps += (spatial_dimension, )
    steps += (coord_system, )
    steps += (numerical, )
    steps += (simd, )
    steps += (scalar_field_init_data, )
    steps += (adm_bssn_converter, )
    steps += (bssn_spacetime, )
    steps += (boundary_condition, )
    steps += (main_ccode, )

    for step in steps:
        step.build()

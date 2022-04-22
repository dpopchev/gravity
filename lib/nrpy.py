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
import BSSN.BSSN_RHSs as rhs
import BSSN.BSSN_gauge_RHSs as gaugerhs
import BSSN.BSSN_quantities as Bq
import ScalarField.ScalarField_RHSs as sfrhs
import ScalarField.ScalarField_Tmunu as sfTmunu
import BSSN.BSSN_stress_energy_source_terms as Bsest
import BSSN.BSSN_constraints as bssncon
import BSSN.Enforce_Detgammahat_Constraint as EGC

import time
from itertools import product
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

@dataclass
class InitialDataConverter:
    """Convert ADM initial data into BSSN-in-curvilinear coordinates"""
    coord_system: CoordSystem = None
    ccodesdir: CcodesDir = None
    adm_input_function_name: str = 'ID_scalarfield_ADM_quantities'
    loopopts: str = ''

    def build(self):
        AtoBnum.Convert_Spherical_or_Cartesian_ADM_to_BSSN_curvilinear(self.coord_system.name,
                                                                       self.adm_input_function_name,
                                                                       Ccodesdir=self.ccodesdir.root,
                                                                       loopopts=self.loopopts)

@dataclass
class BssnRhsBuilder:
    derivatives: Derivatives = None
    ccodesdir: CcodesDir = None
    spatial_dimension: SpatialDimension = None
    rfm_files_dirname: InitVar[str] = 'rfm_files'
    rfm_files_dir: str = None
    is_rfm_precompute_enabled: str = 'True'
    t4uu: str = None
    beta_u: str = None
    detg_constraint_symb_expressions: str = None
    hamiltonian: str = None

    def __post_init__(self, rfm_files_dirname):
        if self.rfm_files_dir is None:
            self.rfm_files_dir = os.path.join(self.ccodesdir.root,
                                              rfm_files_dirname)

    def build_bssn_gauge_rhs(self):
        par.set_parval_from_str("BSSN.BSSN_gauge_RHSs::LapseEvolutionOption", self.derivatives.lapse_condition)
        par.set_parval_from_str("BSSN.BSSN_gauge_RHSs::ShiftEvolutionOption", self.derivatives.shift_condition)

    def build_reference_metric(self):
        cmd.mkdir(self.rfm_files_dir)
        par.set_parval_from_str(
            "reference_metric::enable_rfm_precompute",
            self.is_rfm_precompute_enabled
        )
        par.set_parval_from_str(
            "reference_metric::rfm_precompute_Ccode_outdir",
            self.rfm_files_dir)

    def evaluate_rhs_with_rfm_precomptue(self):
        if self.is_rfm_precompute_enabled != 'True':
            raise AttributeError('not tested scenario, fix and work in different branch')

        par.set_parval_from_str("BSSN.BSSN_quantities::LeaveRicciSymbolic", "True")

        rhs.BSSN_RHSs()

    def evaluate_scalar_field_rhs(self):
        sfrhs.ScalarField_RHSs()

    def compute_scalar_field_energy_momentum_tensor(self):
        if self.t4uu is not None:
            raise AttributeError('Scalar Field Tuu should be computed here')

        sfTmunu.ScalarField_Tmunu()
        self.t4uu = sfTmunu.T4UU

    def compute_bssn_stress_energy_source_terms(self):
        Bsest.BSSN_source_terms_for_BSSN_RHSs(self.t4uu)
        rhs.trK_rhs += Bsest.sourceterm_trK_rhs
        for i in range(self.spatial_dimension.dim):
            # Needed for Gamma-driving shift RHSs:
            rhs.Lambdabar_rhsU[i] += Bsest.sourceterm_Lambdabar_rhsU[i]
            # Needed for BSSN RHSs:
            rhs.lambda_rhsU[i]    += Bsest.sourceterm_lambda_rhsU[i]
            for j in range(self.spatial_dimension.dim):
                rhs.a_rhsDD[i][j] += Bsest.sourceterm_a_rhsDD[i][j]

        gaugerhs.BSSN_gauge_RHSs()
        Bq.BSSN_basic_tensors()
        self.beta_u = Bq.betaU

    def enforce_detgammahat_constraint(self):
        self.detg_constraint_symb_expressions = EGC.Enforce_Detgammahat_Constraint_symb_expressions()

    def compute_ricci_tensor(self):
        par.set_parval_from_str("BSSN.BSSN_quantities::LeaveRicciSymbolic","False")
        Bq.RicciBar__gammabarDD_dHatD__DGammaUDD__DGammaU()

    def build_hamiltonian_gridfunction(self):
        self.hamiltonian = gri.register_gridfunctions("AUX","H")

    def build_hamiltonian_constraint(self):
        bssncon.BSSN_constraints(add_T4UUmunu_source_terms=False)
        Bsest.BSSN_source_terms_for_BSSN_constraints(self.t4uu)
        bssncon.H += Bsest.sourceterm_H

    def build_kreiss_olider_dissipation(self):
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
        indexes = range(3)
        for k, i, j in product(indexes, indexes, indexes):
            gaugerhs.alpha_rhs += diss_strength*alpha_dKOD[k]*rfm.ReU[k]
            rhs.cf_rhs         += diss_strength*   cf_dKOD[k]*rfm.ReU[k]
            rhs.trK_rhs        += diss_strength*  trK_dKOD[k]*rfm.ReU[k]
            sfrhs.sf_rhs       += diss_strength*   sf_dKOD[k]*rfm.ReU[k]
            sfrhs.sfM_rhs      += diss_strength*  sfM_dKOD[k]*rfm.ReU[k]

            if "2ndOrder" in self.derivatives.shift_condition:
                gaugerhs.bet_rhsU[i] += diss_strength*   betU_dKOD[i][k]*rfm.ReU[k]

            gaugerhs.vet_rhsU[i]     += diss_strength*   vetU_dKOD[i][k]*rfm.ReU[k]
            rhs.lambda_rhsU[i]       += diss_strength*lambdaU_dKOD[i][k]*rfm.ReU[k]

            rhs.a_rhsDD[i][j] += diss_strength*aDD_dKOD[i][j][k]*rfm.ReU[k]
            rhs.h_rhsDD[i][j] += diss_strength*hDD_dKOD[i][j][k]*rfm.ReU[k]

    def build_rfm_closed_form_expressions(self):
        par.set_parval_from_str("reference_metric::enable_rfm_precompute","False")
        rfm.ref_metric__hatted_quantities()

    def build_bssn_plus_scalarfield_rhss_c_code(self):
        print("Generating C code for BSSN RHSs in "+par.parval_from_str("reference_metric::CoordSystem")+" coordinates.")
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
        outCfunction(
            outfile  = os.path.join(self.ccodesdir.root,name+".h"), desc=desc, name=name,
            params   = """rfm_struct *restrict rfmstruct,const paramstruct *restrict params,
                          const REAL *restrict auxevol_gfs,const REAL *restrict in_gfs,REAL *restrict rhs_gfs""",
            body     = fin.FD_outputC("returnstring",BSSN_evol_rhss, params="outCverbose=False,enable_SIMD=True",
                                      upwindcontrolvec=self.beta_u),
            loopopts = "InteriorPoints,enable_SIMD,enable_rfm_precompute")
        end = time.time()
        print("(BENCH) Finished BSSN_RHS C codegen in " + str(end - start) + "seconds.")

    def build_ricci_c_code(self):
        print("Generating C code for Ricci tensor in "+par.parval_from_str("reference_metric::CoordSystem")+" coordinates.")
        start = time.time()
        desc="Evaluate the Ricci tensor"
        name="Ricci_eval"
        outCfunction(
            outfile  = os.path.join(self.ccodesdir.root,name+".h"), desc=desc, name=name,
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

    def build_hamiltonian_c_code(self):
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

    def build_gammadet_c_code(self):
        start = time.time()
        print("Generating optimized C code for gamma constraint. May take a while, depending on CoordSystem.")

        # Set up the C function for the det(gammahat) = det(gammabar)
        EGC.output_Enforce_Detgammahat_Constraint_Ccode(Ccodesdir,exprs=enforce_detg_constraint_symb_expressions)
        end = time.time()
        print("(BENCH) Finished gamma constraint C codegen in " + str(end - start) + " seconds.")

    def build(self):
        print("Generating symbolic expressions for BSSN RHSs...")
        start = time.time()
        self.build_bssn_gauge_rhs()
        self.build_reference_metric()
        self.evaluate_rhs_with_rfm_precomptue()
        self.evaluate_scalar_field_rhs()
        self.compute_scalar_field_energy_momentum_tensor()
        self.compute_bssn_stress_energy_source_terms()
        self.enforce_detgammahat_constraint()
        self.compute_ricci_tensor()
        self.build_hamiltonian_gridfunction()
        self.build_hamiltonian_constraint()
        self.build_kreiss_olider_dissipation()
        self.build_rfm_closed_form_expressions()
        end = time.time()
        print("(BENCH) Finished BSSN symbolic expressions in "+str(end-start)+" seconds.")
        self.build_bssn_plus_scalarfield_rhss_c_code()
        self.build_ricci_c_code()
        self.build_hamiltonian_c_code()
        self.build_gammadet_c_code()

def build_scalar_field_collapse():
    ccodesdir = CcodesDir()
    spatial_dimension = SpatialDimension()
    derivatives = Derivatives()
    coord_system = CoordSystem(derivatives=derivatives)
    moltimestepping = RungeKuttaTimesteppingCode(derivatives=derivatives,
                                                 ccodesdir=ccodesdir)
    simd = Simd(ccodesdir=ccodesdir)
    sfinitdata = ScalarFieldInitialData(ccodesdir=ccodesdir, coord_system=coord_system)
    adm_bssn_initial_data_converter = InitialDataConverter(coord_system=coord_system, ccodesdir=ccodesdir)
    bssn_rhs = BssnRhsBuilder(derivatives=derivatives,
                              ccodesdir = ccodesdir,
                              spatial_dimension = spatial_dimension,
                              )

    steps = ( ccodesdir, )
    steps += (spatial_dimension, )
    steps += (moltimestepping, )
    steps += (coord_system, )
    steps += (simd, )
    steps += (sfinitdata, )
    steps += (adm_bssn_initial_data_converter, )
    steps += (bssn_rhs,)

    for step in steps:
        step.build()

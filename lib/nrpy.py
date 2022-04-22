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
import CurviBoundaryConditions.CurviBoundaryConditions as cbcs

import time
from itertools import product
from dataclasses import dataclass, InitVar

import pdb

MAIN_CFILE_NAME = 'ScalarFieldCollapse_Playground.c'
MAIN_CFILE_CONTENT = """
// Step P0: Define REAL and NGHOSTS; and declare CFL_FACTOR. This header is generated in NRPy+.
#include "ScalarFieldCollapse_Playground_REAL__NGHOSTS__CFL_FACTOR.h"

#include "rfm_files/rfm_struct__declare.h"

#include "declare_Cparameters_struct.h"

// All SIMD intrinsics used in SIMD-enabled C code loops are defined here:
#include "SIMD/SIMD_intrinsics.h"

// Step P1: Import needed header files
#include "stdio.h"
#include "stdlib.h"
#include "math.h"
#include "time.h"
#include "stdint.h" // Needed for Windows GCC 6.x compatibility
#ifndef M_PI
#define M_PI 3.141592653589793238462643383279502884L
#endif
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.707106781186547524400844362104849039L
#endif
#define wavespeed 1.0 // Set CFL-based "wavespeed" to 1.0.
#define alpha_threshold (2e-3) // Value below which we rule gravitational collapse has happened

// Step P2: Declare the IDX4S(gf,i,j,k) macro, which enables us to store 4-dimensions of
//           data in a 1D array. In this case, consecutive values of "i"
//           (all other indices held to a fixed value) are consecutive in memory, where
//           consecutive values of "j" (fixing all other indices) are separated by
//           Nxx_plus_2NGHOSTS0 elements in memory. Similarly, consecutive values of
//           "k" are separated by Nxx_plus_2NGHOSTS0*Nxx_plus_2NGHOSTS1 in memory, etc.
#define IDX4S(g,i,j,k) \
( (i) + Nxx_plus_2NGHOSTS0 * ( (j) + Nxx_plus_2NGHOSTS1 * ( (k) + Nxx_plus_2NGHOSTS2 * (g) ) ) )
#define IDX4ptS(g,idx) ( (idx) + (Nxx_plus_2NGHOSTS0*Nxx_plus_2NGHOSTS1*Nxx_plus_2NGHOSTS2) * (g) )
#define IDX3S(i,j,k) ( (i) + Nxx_plus_2NGHOSTS0 * ( (j) + Nxx_plus_2NGHOSTS1 * ( (k) ) ) )
#define LOOP_REGION(i0min,i0max, i1min,i1max, i2min,i2max) \
  for(int i2=i2min;i2<i2max;i2++) for(int i1=i1min;i1<i1max;i1++) for(int i0=i0min;i0<i0max;i0++)
#define LOOP_ALL_GFS_GPS(ii) _Pragma("omp parallel for") \
  for(int (ii)=0;(ii)<Nxx_plus_2NGHOSTS_tot*NUM_EVOL_GFS;(ii)++)

// Step P3: Set UUGF and VVGF macros, as well as xx_to_Cart()
#include "boundary_conditions/gridfunction_defines.h"

// Step P4: Set xx_to_Cart(const paramstruct *restrict params,
//                     REAL *restrict xx[3],
//                     const int i0,const int i1,const int i2,
//                     REAL xCart[3]),
//           which maps xx->Cartesian via
//    {xx[0][i0],xx[1][i1],xx[2][i2]}->{xCart[0],xCart[1],xCart[2]}
#include "xx_to_Cart.h"

// Step P5: Defines set_Nxx_dxx_invdx_params__and__xx(const int EigenCoord, const int Nxx[3],
//                                       paramstruct *restrict params, REAL *restrict xx[3]),
//          which sets params Nxx,Nxx_plus_2NGHOSTS,dxx,invdx, and xx[] for
//          the chosen Eigen-CoordSystem if EigenCoord==1, or
//          CoordSystem if EigenCoord==0.
#include "set_Nxx_dxx_invdx_params__and__xx.h"

// Step P6: Include basic functions needed to impose curvilinear
//          parity and boundary conditions.
#include "boundary_conditions/CurviBC_include_Cfunctions.h"

// Step P7: Implement the algorithm for upwinding.
//          *NOTE*: This upwinding is backwards from
//          usual upwinding algorithms, because the
//          upwinding control vector in BSSN (the shift)
//          acts like a *negative* velocity.
//#define UPWIND_ALG(UpwindVecU) UpwindVecU > 0.0 ? 1.0 : 0.0

// Step P8: Include function for enforcing detgammabar constraint.
#include "enforce_detgammahat_constraint.h"

// Step P9: Find the CFL-constrained timestep
#include "find_timestep.h"

// Step P10: Declare initial data input struct:
//           stores data from initial data solver,
//           so they can be put on the numerical grid.
typedef struct __ID_inputs {
    int interp_stencil_size;
    int numlines_in_file;
    REAL *r_arr,*sf_arr,*psi4_arr,*alpha_arr;
} ID_inputs;

// Part P11: Declare all functions for setting up ScalarField initial data.
/* Routines to interpolate the ScalarField solution and convert to ADM & T^{munu}: */
#include "../ScalarField/ScalarField_interp.h"
#include "ID_scalarfield_ADM_quantities.h"
#include "ID_scalarfield_spherical.h"
#include "ID_scalarfield_xx0xx1xx2_to_BSSN_xx0xx1xx2.h"
#include "ID_scalarfield.h"

/* Next perform the basis conversion and compute all needed BSSN quantities */
#include "ID_ADM_xx0xx1xx2_to_BSSN_xx0xx1xx2__ALL_BUT_LAMBDAs.h"
#include "ID_BSSN__ALL_BUT_LAMBDAs.h"
#include "ID_BSSN_lambdas.h"

// Step P12: Set the generic driver function for setting up BSSN initial data
void initial_data(const paramstruct *restrict params,const bc_struct *restrict bcstruct,
                  const rfm_struct *restrict rfmstruct,
                  REAL *restrict xx[3], REAL *restrict auxevol_gfs, REAL *restrict in_gfs) {
#include "set_Cparameters.h"

    // Step 1: Set up ScalarField initial data
    // Step 1.a: Read ScalarField initial data from data file
    // Open the data file:
    char filename[100];
    sprintf(filename,"./SFID.txt");
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        fprintf(stderr,"ERROR: could not open file %s\n",filename);
        exit(1);
    }
    // Count the number of lines in the data file:
    int numlines_in_file = count_num_lines_in_file(fp);
    // Allocate space for all data arrays:
    REAL *r_arr     = (REAL *)malloc(sizeof(REAL)*numlines_in_file);
    REAL *sf_arr    = (REAL *)malloc(sizeof(REAL)*numlines_in_file);
    REAL *psi4_arr  = (REAL *)malloc(sizeof(REAL)*numlines_in_file);
    REAL *alpha_arr = (REAL *)malloc(sizeof(REAL)*numlines_in_file);

    // Read from the data file, filling in arrays
    // read_datafile__set_arrays() may be found in ScalarField/ScalarField_interp.h
    if(read_datafile__set_arrays(fp,r_arr,sf_arr,psi4_arr,alpha_arr) == 1) {
        fprintf(stderr,"ERROR WHEN READING FILE %s!\n",filename);
        exit(1);
    }
    fclose(fp);

    const int interp_stencil_size = 12;
    ID_inputs SF_in;
    SF_in.interp_stencil_size = interp_stencil_size;
    SF_in.numlines_in_file    = numlines_in_file;
    SF_in.r_arr               = r_arr;
    SF_in.sf_arr              = sf_arr;
    SF_in.psi4_arr            = psi4_arr;
    SF_in.alpha_arr           = alpha_arr;

    // Step 1.b: Interpolate data from data file to set BSSN gridfunctions
    ID_scalarfield(params,xx,SF_in, in_gfs);
    ID_BSSN__ALL_BUT_LAMBDAs(params,xx,SF_in, in_gfs);
    apply_bcs_curvilinear(params, bcstruct, NUM_EVOL_GFS, evol_gf_parity, in_gfs);
    enforce_detgammahat_constraint(rfmstruct, params,                   in_gfs);
    ID_BSSN_lambdas(params, xx, in_gfs);
    apply_bcs_curvilinear(params, bcstruct, NUM_EVOL_GFS, evol_gf_parity, in_gfs);
    enforce_detgammahat_constraint(rfmstruct, params,                   in_gfs);

    free(r_arr);
    free(sf_arr);
    free(psi4_arr);
    free(alpha_arr);
}

// Step P11: Declare function for evaluating Hamiltonian constraint (diagnostic)
#include "Hamiltonian_constraint.h"

// Step P12: Declare rhs_eval function, which evaluates BSSN RHSs
#include "rhs_eval.h"

// Step P13: Declare Ricci_eval function, which evaluates Ricci tensor
#include "Ricci_eval.h"

//#include "NRPyCritCol_regridding.h"

REAL rho_max = 0.0;

// main() function:
// Step 0: Read command-line input, set up grid structure, allocate memory for gridfunctions, set up coordinates
// Step 1: Set up initial data to an exact solution
// Step 2: Start the timer, for keeping track of how fast the simulation is progressing.
// Step 3: Integrate the initial data forward in time using the chosen RK-like Method of
//         Lines timestepping algorithm, and output periodic simulation diagnostics
// Step 3.a: Output 2D data file periodically, for visualization
// Step 3.b: Step forward one timestep (t -> t+dt) in time using
//           chosen RK-like MoL timestepping algorithm
// Step 3.c: If t=t_final, output conformal factor & Hamiltonian
//           constraint violation to 1D data file
// Step 3.d: Progress indicator printing to stderr
// Step 4: Free all allocated memory
int main(int argc, const char *argv[]) {
    paramstruct params;
#include "set_Cparameters_default.h"

    // Step 0a: Read command-line input, error out if nonconformant
    if((argc != 4 && argc != 5) || atoi(argv[1]) < NGHOSTS || atoi(argv[2]) < 2 || atoi(argv[3]) < 2 /* FIXME; allow for axisymmetric sims */) {
        fprintf(stderr,"Error: Expected three command-line arguments: ./ScalarFieldCollapse_Playground Nx0 Nx1 Nx2,\n");
        fprintf(stderr,"where Nx[0,1,2] is the number of grid points in the 0, 1, and 2 directions.\n");
        fprintf(stderr,"Nx[] MUST BE larger than NGHOSTS (= %d)\n",NGHOSTS);
        exit(1);
    }
    if(argc == 5) {
        CFL_FACTOR = strtod(argv[4],NULL);
        if(CFL_FACTOR > 0.5 && atoi(argv[3])!=2) {
            fprintf(stderr,"WARNING: CFL_FACTOR was set to %e, which is > 0.5.\n",CFL_FACTOR);
            fprintf(stderr,"         This will generally only be stable if the simulation is purely axisymmetric\n");
            fprintf(stderr,"         However, Nx2 was set to %d>2, which implies a non-axisymmetric simulation\n",atoi(argv[3]));
        }
    }
    // Step 0b: Set up numerical grid structure, first in space...
    const int Nxx[3] = { atoi(argv[1]), atoi(argv[2]), atoi(argv[3]) };
    if(Nxx[0]%2 != 0 || Nxx[1]%2 != 0 || Nxx[2]%2 != 0) {
        fprintf(stderr,"Error: Cannot guarantee a proper cell-centered grid if number of grid cells not set to even number.\n");
        fprintf(stderr,"       For example, in case of angular directions, proper symmetry zones will not exist.\n");
        exit(1);
    }

    // Step 0c: Set free parameters, overwriting Cparameters defaults
    //          by hand or with command-line input, as desired.
#include "free_parameters.h"

   // Step 0d: Uniform coordinate grids are stored to *xx[3]
    REAL *xx[3];
    // Step 0d.i: Set bcstruct
    bc_struct bcstruct;
    {
        int EigenCoord = 1;
        // Step 0d.ii: Call set_Nxx_dxx_invdx_params__and__xx(), which sets
        //             params Nxx,Nxx_plus_2NGHOSTS,dxx,invdx, and xx[] for the
        //             chosen Eigen-CoordSystem.
        set_Nxx_dxx_invdx_params__and__xx(EigenCoord, Nxx, &params, xx);
        // Step 0d.iii: Set Nxx_plus_2NGHOSTS_tot
#include "set_Cparameters-nopointer.h"
        const int Nxx_plus_2NGHOSTS_tot = Nxx_plus_2NGHOSTS0*Nxx_plus_2NGHOSTS1*Nxx_plus_2NGHOSTS2;
        // Step 0e: Find ghostzone mappings; set up bcstruct
#include "boundary_conditions/driver_bcstruct.h"
        // Step 0e.i: Free allocated space for xx[][] array
        for(int i=0;i<3;i++) free(xx[i]);
    }

    // Step 0f: Call set_Nxx_dxx_invdx_params__and__xx(), which sets
    //          params Nxx,Nxx_plus_2NGHOSTS,dxx,invdx, and xx[] for the
    //          chosen (non-Eigen) CoordSystem.
    int EigenCoord = 0;
    set_Nxx_dxx_invdx_params__and__xx(EigenCoord, Nxx, &params, xx);

    // Step 0g: Set all C parameters "blah" for params.blah, including
    //          Nxx_plus_2NGHOSTS0 = params.Nxx_plus_2NGHOSTS0, etc.
#include "set_Cparameters-nopointer.h"
    const int Nxx_plus_2NGHOSTS_tot = Nxx_plus_2NGHOSTS0*Nxx_plus_2NGHOSTS1*Nxx_plus_2NGHOSTS2;

    // Step 0h: Time coordinate parameters
    REAL t_final = 16.0; /* Final time is set so that at t=t_final,
                          * data at the origin have not been corrupted
                          * by the approximate outer boundary condition */

    // Step 0i: Set timestep based on smallest proper distance between gridpoints and CFL factor
    REAL dt = find_timestep(&params, xx);
    //fprintf(stderr,"# Timestep set to = %e\n",(double)dt);
    int N_final = (int)(t_final / dt + 0.5); // The number of points in time.
                                             // Add 0.5 to account for C rounding down
                                             // typecasts to integers.
    int output_every_N = 20;//(int)((REAL)N_final/800.0);
    if(output_every_N == 0) output_every_N = 1;

    // Step 0j: Error out if the number of auxiliary gridfunctions outnumber evolved gridfunctions.
    //              This is a limitation of the RK method. You are always welcome to declare & allocate
    //              additional gridfunctions by hand.
    if(NUM_AUX_GFS > NUM_EVOL_GFS) {
        fprintf(stderr,"Error: NUM_AUX_GFS > NUM_EVOL_GFS. Either reduce the number of auxiliary gridfunctions,\n");
        fprintf(stderr,"       or allocate (malloc) by hand storage for *diagnostic_output_gfs. \n");
        exit(1);
    }

    // Step 0k: Allocate memory for gridfunctions
#include "MoLtimestepping/RK_Allocate_Memory.h"
    REAL *restrict auxevol_gfs = (REAL *)malloc(sizeof(REAL) * NUM_AUXEVOL_GFS * Nxx_plus_2NGHOSTS_tot);

    // Step 0l: Set up precomputed reference metric arrays
    // Step 0l.i: Allocate space for precomputed reference metric arrays.
#include "rfm_files/rfm_struct__malloc.h"

    // Step 0l.ii: Define precomputed reference metric arrays.
    {
    #include "set_Cparameters-nopointer.h"
    #include "rfm_files/rfm_struct__define.h"
    }

    // Step 1: Set up initial data to an exact solution
    initial_data(&params,&bcstruct, &rfmstruct, xx, auxevol_gfs, y_n_gfs);

    // Step 1b: Apply boundary conditions, as initial data
    //          are sometimes ill-defined in ghost zones.
    //          E.g., spherical initial data might not be
    //          properly defined at points where r=-1.
    apply_bcs_curvilinear(&params, &bcstruct, NUM_EVOL_GFS,evol_gf_parity, y_n_gfs);
    enforce_detgammahat_constraint(&rfmstruct, &params, y_n_gfs);

    // Step 2: Start the timer, for keeping track of how fast the simulation is progressing.
#ifdef __linux__ // Use high-precision timer in Linux.
    struct timespec start, end;
    clock_gettime(CLOCK_REALTIME, &start);
#else     // Resort to low-resolution, standards-compliant timer in non-Linux OSs
    // http://www.cplusplus.com/reference/ctime/time/
    time_t start_timer,end_timer;
    time(&start_timer); // Resolution of one second...
#endif

    // Step 3: Integrate the initial data forward in time using the chosen RK-like Method of
    //         Lines timestepping algorithm, and output periodic simulation diagnostics
    for(int n=0;n<=N_final;n++) { // Main loop to progress forward in time.

        // Step 3.a: Output 2D data file periodically, for visualization
        if(n%output_every_N == 0) {
            // Evaluate Hamiltonian constraint violation
            Hamiltonian_constraint(&rfmstruct, &params, y_n_gfs,auxevol_gfs, diagnostic_output_gfs);

            char filename[100];
            sprintf(filename,"out%d-%08d.txt",Nxx[0],n);
            const int i1mid=Nxx_plus_2NGHOSTS1/2;
            const int i2mid=Nxx_plus_2NGHOSTS2/2;
            FILE *fp = fopen(filename, "w");
            for( int i0=NGHOSTS;i0<Nxx_plus_2NGHOSTS0-NGHOSTS;i0++) {
                const int idx  = IDX3S(i0,i1mid,i2mid);
                const REAL xx0 = xx[0][i0];
                REAL xCart[3];
                xx_to_Cart(&params,xx,i0,i1mid,i2mid,xCart);
                const REAL rr = sqrt( xCart[0]*xCart[0] + xCart[1]*xCart[1] + xCart[2]*xCart[2] );
                fprintf(fp,"%e %e %e %e %e %e %e\n",xx0,rr,
                        y_n_gfs[IDX4ptS(SFGF,idx)],y_n_gfs[IDX4ptS(SFMGF,idx)],
                        y_n_gfs[IDX4ptS(ALPHAGF,idx)],y_n_gfs[IDX4ptS(CFGF,idx)],
                        log10(fabs(diagnostic_output_gfs[IDX4ptS(HGF,idx)])));
            }
            fclose(fp);
        }

        // Step 3.b: Step forward one timestep (t -> t+dt) in time using
        //           chosen RK-like MoL timestepping algorithm
#include "MoLtimestepping/RK_MoL.h"

        // Step 3.c: If t=t_final, output conformal factor & Hamiltonian
        //           constraint violation to 2D data file
        if(n==N_final-1) {
            // Evaluate Hamiltonian constraint violation
            Hamiltonian_constraint(&rfmstruct, &params, y_n_gfs,auxevol_gfs, diagnostic_output_gfs);
            char filename[100];
            sprintf(filename,"out%d.txt",Nxx[0]);
            FILE *out1D = fopen(filename, "w");
            const int i1mid=Nxx_plus_2NGHOSTS1/2;
            const int i2mid=Nxx_plus_2NGHOSTS2/2;
            for(int i0=NGHOSTS;i0<Nxx_plus_2NGHOSTS0-NGHOSTS;i0++) {
                REAL xCart[3];
                xx_to_Cart(&params,xx,i0,i1mid,i2mid,xCart);
                const REAL rr = sqrt( xCart[0]*xCart[0] + xCart[1]*xCart[1] + xCart[2]*xCart[2] );
                int idx = IDX3S(i0,i1mid,i2mid);
                fprintf(out1D,"%e %e\n",rr,log10(fabs(diagnostic_output_gfs[IDX4ptS(HGF,idx)])));
            }
            fclose(out1D);
        }

        // Step 3.d: Progress indicator printing to stderr

        // Step 3.d.i: Measure average time per iteration
#ifdef __linux__ // Use high-precision timer in Linux.
        clock_gettime(CLOCK_REALTIME, &end);
        const long long unsigned int time_in_ns = 1000000000L * (end.tv_sec - start.tv_sec) + end.tv_nsec - start.tv_nsec;
#else     // Resort to low-resolution, standards-compliant timer in non-Linux OSs
        time(&end_timer); // Resolution of one second...
        REAL time_in_ns = difftime(end_timer,start_timer)*1.0e9+0.5; // Round up to avoid divide-by-zero.
#endif
        const REAL s_per_iteration_avg = ((REAL)time_in_ns / (REAL)n) / 1.0e9;

        const int iterations_remaining = N_final - n;
        const REAL time_remaining_in_mins = s_per_iteration_avg * (REAL)iterations_remaining / 60.0;

        const REAL num_RHS_pt_evals = (REAL)(Nxx[0]*Nxx[1]*Nxx[2]) * 4.0 * (REAL)n; // 4 RHS evals per gridpoint for RK4
        const REAL RHS_pt_evals_per_sec = num_RHS_pt_evals / ((REAL)time_in_ns / 1.0e9);


        // Step 3.d.ii: Output simulation progress to stderr
        if(n%10 == 0) {
            fprintf(stderr,"%c[2K", 27); // Clear the line
            fprintf(stderr,"It: %d t=%.2f dt=%.2e | %.1f%%; ETA %.0f s | t/h %.2f | gp/s %.2e\r",  // \r is carriage return, move cursor to the beginning of the line
                   n, n*dt, (double)dt, (double)(100.0 * (REAL)n / (REAL)N_final),
                   (double)time_remaining_in_mins*60, (double)(dt * 3600.0 / s_per_iteration_avg), (double)RHS_pt_evals_per_sec);
            fflush(stderr); // Flush the stderr buffer
        } // End progress indicator if(n % 10 == 0)
    } // End main loop to progress forward in time.
    fprintf(stderr,"\n"); // Clear the final line of output from progress indicator.

    // Step 4: Free all allocated memory
#include "rfm_files/rfm_struct__freemem.h"
#include "boundary_conditions/bcstruct_freemem.h"
#include "MoLtimestepping/RK_Free_Memory.h"
    free(auxevol_gfs);
    for(int i=0;i<3;i++) free(xx[i]);

    return 0;
}
"""

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
    coord_system: CoordSystem = None
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
        EGC.output_Enforce_Detgammahat_Constraint_Ccode(self.ccodesdir.root,
                                                        exprs=self.detg_constraint_symb_expressions)
        end = time.time()
        print("(BENCH) Finished gamma constraint C codegen in " + str(end - start) + " seconds.")

    def build_c_code_declaring_and_setting_cparameters(self):
        # Step 4.e.i: Generate declare_Cparameters_struct.h, set_Cparameters_default.h, and set_Cparameters[-SIMD].h
        par.generate_Cparameters_Ccodes(os.path.join(self.ccodesdir.root))

        # Step 4.e.ii: Set free_parameters.h
        # Output to $Ccodesdir/free_parameters.h reference metric parameters based on generic
        #    domain_size,sinh_width,sinhv2_const_dr,SymTP_bScale,
        #    parameters set above.
        rfm.out_default_free_parameters_for_rfm(os.path.join(self.ccodesdir.root,"free_parameters.h"),
                                                self.coord_system.domain_size,
                                                self.coord_system.sinh_width,
                                                self.coord_system.sinhv2_const_dr,
                                                self.coord_system.symtp_bscale
                                                )

        # Step 4.e.iii: Generate set_Nxx_dxx_invdx_params__and__xx.h:
        rfm.set_Nxx_dxx_invdx_params__and__xx_h(self.ccodesdir.root)

        # Step 4.e.iv: Generate xx_to_Cart.h, which contains xx_to_Cart() for
        #               (the mapping from xx->Cartesian) for the chosen
        #               CoordSystem:
        rfm.xx_to_Cart_h("xx_to_Cart",
                         "./set_Cparameters.h",
                         os.path.join(self.ccodesdir.root,"xx_to_Cart.h"))

        # Step 4.e.v: Generate declare_Cparameters_struct.h, set_Cparameters_default.h, and set_Cparameters[-SIMD].h
        par.generate_Cparameters_Ccodes(os.path.join(self.ccodesdir.root))

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
        self.build_c_code_declaring_and_setting_cparameters()

@dataclass
class BoundaryConditionFunctions:
    ccodesdir: CcodesDir = None

    def build(self):
        cbcs.Set_up_CurviBoundaryConditions(os.path.join(self.ccodesdir.root,"boundary_conditions/"),
                                            Cparamspath=os.path.join("../"),
                                            path_prefix='../nrpytutorial')

@dataclass
class MainCcode:
    ccodesdir: CcodesDir = None
    derivatives: Derivatives = None

    def build(self):
        fpath = os.path.join(self.ccodesdir.root,"ScalarFieldCollapse_Playground_REAL__NGHOSTS__CFL_FACTOR.h")

        fcontent = """
// Part P0.a: Set the number of ghost cells, from NRPy+'s FD_CENTDERIVS_ORDER
#define NGHOSTS """+str(int(self.derivatives.fd_order/2)+1)+"""
// Part P0.b: Set the numerical precision (REAL) to double, ensuring all floating point
//            numbers are stored to at least ~16 significant digits
#define REAL """+self.derivatives.real+"""
// Part P0.c: Set the number of ghost cells, from NRPy+'s FD_CENTDERIVS_ORDER
REAL CFL_FACTOR = """+str(self.derivatives.cfl_factor)+"""; // Set the CFL Factor. Can be overwritten at command line.
"""
        with open(fpath, 'w') as fhandler:
            fhandler.write(fcontent)

        fpath = os.path.join(self.ccodesdir.root,MAIN_CFILE_NAME)
        with open(fpath, 'w') as fhandler:
            fhandler.write(MAIN_CFILE_CONTENT)

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
                              coord_system = coord_system
                              )
    boundary_conditions_functions = BoundaryConditionFunctions(ccodesdir=ccodesdir)
    mainccode = MainCcode(ccodesdir=ccodesdir, derivatives=derivatives)

    steps = ( ccodesdir, )
    steps += (spatial_dimension, )
    steps += (moltimestepping, )
    steps += (coord_system, )
    steps += (simd, )
    steps += (sfinitdata, )
    steps += (adm_bssn_initial_data_converter, )
    steps += (bssn_rhs,)
    steps += (boundary_conditions_functions,)
    steps += (mainccode,)

    for step in steps:
        step.build()

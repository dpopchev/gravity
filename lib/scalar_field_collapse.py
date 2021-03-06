import os
import adapters
import nrpy_local as nrpy
from ccode_builders import build_timestepping_ccode_generator
from ccode_builders import build_scalar_field_initial_data_ccode_generator
from ccode_builders import build_param_funcs_basic_defines_scalar_field
from ccode_builders import build_converter_adm_bssn_init_data
from ccode_builders import build_scalar_field_collapse_playground_header
from ccode_builders import build_scalar_field_collapse_playground_main
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
from matplotlib.pyplot import savefig
import glob
import sys

def visualize_all(ccodes_dir):
    globby = glob.glob(os.path.join(ccodes_dir.outdir,'out640-00*.txt'))
    file_list = []
    for x in sorted(globby):
        file_list.append(x)

    for filename in file_list:
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

def visualize_conv(ccodes_dir, numerical_integration):
    os.chdir(ccodes_dir.outdir)

    nrpy.cmd.delete_existing_files("out320*.txt")
    nrpy.cmd.Execute("ScalarFieldCollapse_Playground", "320 2 2 "+str(numerical_integration.cfl_factor),"out320.txt")

    os.chdir(os.path.join("..",".."))

    outfig = os.path.join(ccodes_dir.outdir,"ScalarFieldCollapse_H_convergence.png")

    fig = plt.figure()

    r_640,H_640 = np.loadtxt(os.path.join(ccodes_dir.outdir,"out640.txt")).T
    r_320,H_320 = np.loadtxt(os.path.join(ccodes_dir.outdir,"out320.txt")).T

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
    Image(outfig)

def build_bssn_rhs_symbolic_expressions(ccodes_dir, numerical_integration, dim):
    print("Generating symbolic expressions for BSSN RHSs...")

    # Enable rfm_precompute infrastructure, which results in
    #   BSSN RHSs that are free of transcendental functions,
    #   even in curvilinear coordinates, so long as
    #   ConformalFactor is set to "W" (default).
    enable_rfm_precompute = adapters.InterfaceParameter.build("reference_metric::enable_rfm_precompute","True")

    # Evaluate BSSN + BSSN gauge RHSs with rfm_precompute enabled:
    rfm_files = ccodes_dir.make_under_root('rfm_files')
    rfm_precompute_Ccode_outdir = adapters.InterfaceParameter.build("reference_metric::rfm_precompute_Ccode_outdir", rfm_files)
    leave_ricci_symbolic = adapters.InterfaceParameter.build("BSSN.BSSN_quantities::LeaveRicciSymbolic","True")
    nrpy.rhs.BSSN_RHSs()

    # Evaluate the Scalar Field RHSs
    nrpy.sfrhs.ScalarField_RHSs()

    # Compute ScalarField T^{\mu\nu}
    # Compute the scalar field energy-momentum tensor
    nrpy.sfTmunu.ScalarField_Tmunu()
    scalar_field_contravariant_tmunu = nrpy.sfTmunu.T4UU

    nrpy.Bsest.BSSN_source_terms_for_BSSN_RHSs(scalar_field_contravariant_tmunu)
    nrpy.rhs.trK_rhs += nrpy.Bsest.sourceterm_trK_rhs
    for i in range(dim.representation):
        # Needed for Gamma-driving shift RHSs:
        nrpy.rhs.Lambdabar_rhsU[i] += nrpy.Bsest.sourceterm_Lambdabar_rhsU[i]
        # Needed for BSSN RHSs:
        nrpy.rhs.lambda_rhsU[i]    += nrpy.Bsest.sourceterm_lambda_rhsU[i]
        for j in range(dim.representation):
            nrpy.rhs.a_rhsDD[i][j] += nrpy.Bsest.sourceterm_a_rhsDD[i][j]

    nrpy.gaugerhs.BSSN_gauge_RHSs()
    # We use betaU as our upwinding control vector:
    nrpy.Bq.BSSN_basic_tensors()
    betaU = nrpy.Bq.betaU

    enforce_detg_constraint_symb_expressions = nrpy.EGC.Enforce_Detgammahat_Constraint_symb_expressions()

    # Next compute Ricci tensor
    leave_ricci_symbolic = adapters.InterfaceParameter.build("BSSN.BSSN_quantities::LeaveRicciSymbolic","False")
    nrpy.Bq.RicciBar__gammabarDD_dHatD__DGammaUDD__DGammaU()

    # Now register the Hamiltonian as a gridfunction.
    H = nrpy.gri.register_gridfunctions("AUX","H")

    # Then define the Hamiltonian constraint and output the optimized C code.
    nrpy.bssncon.BSSN_constraints(add_T4UUmunu_source_terms=False)
    nrpy.Bsest.BSSN_source_terms_for_BSSN_constraints(scalar_field_contravariant_tmunu)
    nrpy.bssncon.H += nrpy.Bsest.sourceterm_H

    # Add Kreiss-Oliger dissipation
    diss_strength = adapters.InterfaceCparameter.build( "REAL", "ScalarFieldCollapse", ["diss_strength"], 0.1)

    alpha_dKOD   = nrpy.ixp.declarerank1("alpha_dKOD")
    cf_dKOD      = nrpy.ixp.declarerank1("cf_dKOD")
    trK_dKOD     = nrpy.ixp.declarerank1("trK_dKOD")
    sf_dKOD      = nrpy.ixp.declarerank1("sf_dKOD")
    sfM_dKOD     = nrpy.ixp.declarerank1("sfM_dKOD")
    betU_dKOD    = nrpy.ixp.declarerank2("betU_dKOD","nosym")
    vetU_dKOD    = nrpy.ixp.declarerank2("vetU_dKOD","nosym")
    lambdaU_dKOD = nrpy.ixp.declarerank2("lambdaU_dKOD","nosym")
    aDD_dKOD     = nrpy.ixp.declarerank3("aDD_dKOD","sym01")
    hDD_dKOD     = nrpy.ixp.declarerank3("hDD_dKOD","sym01")

    for k in range(3):
        nrpy.gaugerhs.alpha_rhs += diss_strength.representation*alpha_dKOD[k]*nrpy.rfm.ReU[k]
        nrpy.rhs.cf_rhs         += diss_strength.representation*   cf_dKOD[k]*nrpy.rfm.ReU[k]
        nrpy.rhs.trK_rhs        += diss_strength.representation*  trK_dKOD[k]*nrpy.rfm.ReU[k]
        nrpy.sfrhs.sf_rhs       += diss_strength.representation*   sf_dKOD[k]*nrpy.rfm.ReU[k]
        nrpy.sfrhs.sfM_rhs      += diss_strength.representation*  sfM_dKOD[k]*nrpy.rfm.ReU[k]
        for i in range(3):
            if "2ndOrder" in numerical_integration.shift_condition:
                nrpy.gaugerhs.bet_rhsU[i] += diss_strength.representation*   betU_dKOD[i][k]*nrpy.rfm.ReU[k]
            nrpy.gaugerhs.vet_rhsU[i]     += diss_strength.representation*   vetU_dKOD[i][k]*nrpy.rfm.ReU[k]
            nrpy.rhs.lambda_rhsU[i]       += diss_strength.representation*lambdaU_dKOD[i][k]*nrpy.rfm.ReU[k]
            for j in range(3):
                nrpy.rhs.a_rhsDD[i][j] += diss_strength.representation*aDD_dKOD[i][j][k]*nrpy.rfm.ReU[k]
                nrpy.rhs.h_rhsDD[i][j] += diss_strength.representation*hDD_dKOD[i][j][k]*nrpy.rfm.ReU[k]


    # Now that we are finished with all the rfm hatted
    #           quantities in generic precomputed functional
    #           form, let's restore them to their closed-
    #           form expressions.
    # Reset to False to disable rfm_precompute.
    enable_rfm_precompute = adapters.InterfaceParameter.build("reference_metric::enable_rfm_precompute","False")
    nrpy.rfm.ref_metric__hatted_quantities()
    print("Finished BSSN symbolic expressions")

    return betaU,enforce_detg_constraint_symb_expressions

def build_bssn_plus_scalarfield_rhss(ccodes_dir, betaU):
    rfm_coord_system = nrpy.par.parval_from_str("reference_metric::CoordSystem")
    print(f"Generating C code for BSSN RHSs in {rfm_coord_system} coordinates.")

    # Construct the left-hand sides and right-hand-side expressions for all BSSN RHSs
    lhs_names = [        "alpha",       "cf",       "trK",         "sf",         "sfM"   ]
    rhs_exprs = [nrpy.gaugerhs.alpha_rhs, nrpy.rhs.cf_rhs, nrpy.rhs.trK_rhs, nrpy.sfrhs.sf_rhs, nrpy.sfrhs.sfM_rhs]

    for i in range(3):
        lhs_names.append(        "betU"+str(i))
        rhs_exprs.append(nrpy.gaugerhs.bet_rhsU[i])
        lhs_names.append(   "lambdaU"+str(i))
        rhs_exprs.append(nrpy.rhs.lambda_rhsU[i])
        lhs_names.append(        "vetU"+str(i))
        rhs_exprs.append(nrpy.gaugerhs.vet_rhsU[i])
        for j in range(i,3):
            lhs_names.append(   "aDD"+str(i)+str(j))
            rhs_exprs.append(nrpy.rhs.a_rhsDD[i][j])
            lhs_names.append(   "hDD"+str(i)+str(j))
            rhs_exprs.append(nrpy.rhs.h_rhsDD[i][j])

    # Sort the lhss list alphabetically, and rhss to match.
    #   This ensures the RHSs are evaluated in the same order
    #   they're allocated in memory:
    lhs_names,rhs_exprs = [list(x) for x in zip(*sorted(zip(lhs_names,rhs_exprs), key=lambda pair: pair[0]))]

    # Declare the list of lhrh's
    BSSN_evol_rhss = []
    for var in range(len(lhs_names)):
        BSSN_evol_rhss.append(nrpy.lhrh(lhs=nrpy.gri.gfaccess("rhs_gfs",lhs_names[var]),rhs=rhs_exprs[var]))

    # Set up the C function for the BSSN RHSs
    desc="Evaluate the BSSN RHSs"
    name="rhs_eval"
    outfile_header = ccodes_dir.make_under_root(f'{name}.h', is_dir=False)
    nrpy.outCfunction(
        outfile  = outfile_header,
        desc=desc,
        name=name,
        params   = """rfm_struct *restrict rfmstruct,const paramstruct *restrict params,
                      const REAL *restrict auxevol_gfs,const REAL *restrict in_gfs,REAL *restrict rhs_gfs""",
        body     = nrpy.fin.FD_outputC("returnstring",BSSN_evol_rhss, params="outCverbose=False,enable_SIMD=True",
                                  upwindcontrolvec=betaU),
        loopopts = "InteriorPoints,enable_SIMD,enable_rfm_precompute")

    print("Finished BSSN_RHS C codegen")
    return

def build_ricci(ccodes_dir):
    rfm_coord_system = nrpy.par.parval_from_str("reference_metric::CoordSystem")
    print(f"Generating C code for Ricci tensor in {rfm_coord_system} coordinates.")

    desc="Evaluate the Ricci tensor"
    name="Ricci_eval"
    outfile_header = ccodes_dir.make_under_root(f'{name}.h', is_dir=False)
    nrpy.outCfunction(
        outfile  = outfile_header,
        desc=desc,
        name=name,
        params   = """rfm_struct *restrict rfmstruct,const paramstruct *restrict params,
                      const REAL *restrict in_gfs,REAL *restrict auxevol_gfs""",
        body     = nrpy.fin.FD_outputC("returnstring",
                                  [nrpy.lhrh(lhs=nrpy.gri.gfaccess("auxevol_gfs","RbarDD00"),rhs=nrpy.Bq.RbarDD[0][0]),
                                   nrpy.lhrh(lhs=nrpy.gri.gfaccess("auxevol_gfs","RbarDD01"),rhs=nrpy.Bq.RbarDD[0][1]),
                                   nrpy.lhrh(lhs=nrpy.gri.gfaccess("auxevol_gfs","RbarDD02"),rhs=nrpy.Bq.RbarDD[0][2]),
                                   nrpy.lhrh(lhs=nrpy.gri.gfaccess("auxevol_gfs","RbarDD11"),rhs=nrpy.Bq.RbarDD[1][1]),
                                   nrpy.lhrh(lhs=nrpy.gri.gfaccess("auxevol_gfs","RbarDD12"),rhs=nrpy.Bq.RbarDD[1][2]),
                                   nrpy.lhrh(lhs=nrpy.gri.gfaccess("auxevol_gfs","RbarDD22"),rhs=nrpy.Bq.RbarDD[2][2])],
                                   params="outCverbose=False,enable_SIMD=True"),
        loopopts = "InteriorPoints,enable_SIMD,enable_rfm_precompute")
    print("Finished Ricci C codegen in.")
    return

def build_hamiltonian(ccodes_dir):
    print("Generating optimized C code for Hamiltonian constraint. May take a while, depending on CoordSystem.")
    # Set up the C function for the Hamiltonian RHS
    desc="Evaluate the Hamiltonian constraint"
    name="Hamiltonian_constraint"
    outfile_header = ccodes_dir.make_under_root(f'{name}.h', is_dir=False)
    nrpy.outCfunction(
        outfile  = outfile_header, desc=desc, name=name,
        params   = """rfm_struct *restrict rfmstruct,const paramstruct *restrict params,
                      REAL *restrict in_gfs, REAL *restrict auxevol_gfs, REAL *restrict aux_gfs""",
        body     = nrpy.fin.FD_outputC("returnstring",nrpy.lhrh(lhs=nrpy.gri.gfaccess("aux_gfs", "H"), rhs=nrpy.bssncon.H),
                                  params="outCverbose=False"),
        loopopts = "InteriorPoints,enable_rfm_precompute")
    print("Finished Hamiltonian C codegen.")
    return

def build_cparamters_headers(ccodes_dir, coord_system):
    nrpy.par.generate_Cparameters_Ccodes(ccodes_dir.root)
    free_parameters_header = ccodes_dir.make_under_root("free_parameters.h", is_dir=False)
    nrpy.rfm.out_default_free_parameters_for_rfm(
        free_parameters_header,
        coord_system.domain_size,
        coord_system.sinh_width,
        coord_system.sinhv2_const_dr,
        coord_system.symtp_bscale
    )
    nrpy.rfm.set_Nxx_dxx_invdx_params__and__xx_h(ccodes_dir.root)

    xx_to_cart_h = ccodes_dir.make_under_root("xx_to_Cart.h", is_dir=False)
    nrpy.rfm.xx_to_Cart_h("xx_to_Cart","./set_Cparameters.h",xx_to_cart_h)

    nrpy.par.generate_Cparameters_Ccodes(ccodes_dir.root)

    return

def build_boundary_condition(ccodes_dir):
    boundary_condition = ccodes_dir.make_under_root("boundary_conditions/")

    nrpy.cbcs.Set_up_CurviBoundaryConditions(
        boundary_condition,
        Cparamspath=os.path.join("../"),
        path_prefix='../nrpytutorial')
    return

def build():
    ccodes_dir = adapters.CcodesDir.build()
    dim = adapters.InterfaceParameter.build('grid::DIM', 3)
    coord_system = adapters.CoordSystem.build_spherical()
    numerical_integration = adapters.NumericalIntegration.build()

    timestepping_ccode_generator = build_timestepping_ccode_generator(ccodes_dir, numerical_integration)
    timestepping_ccode_generator.doit()

    reference_metric = adapters.InterfaceParameter.build("reference_metric::CoordSystem", str(coord_system.name))
    nrpy.rfm.reference_metric()

    finite_difference = adapters.InterfaceParameter.build('finite_difference::FD_CENTDERIVS_ORDER', numerical_integration.fd_order)

    simd_src = os.path.join('../nrpytutorial/SIMD/', 'SIMD_intrinsics.h')
    simd_dst = ccodes_dir.make_under_root('SIMD')
    nrpy.shutil.copy(simd_src,simd_dst)

    indexedexp = adapters.InterfaceParameter.build("indexedexp::symmetry_axes", "12")

    find_timestep_header = ccodes_dir.make_under_root('find_timestep.h', is_dir=False)
    nrpy.rfm.out_timestep_func_to_file(find_timestep_header)

    scalar_field_initial_data = build_scalar_field_initial_data_ccode_generator(ccodes_dir, coord_system)
    scalar_field_initial_data.doit()

    param_funcs_basic_defines_scalar_field = build_param_funcs_basic_defines_scalar_field(ccodes_dir)
    param_funcs_basic_defines_scalar_field.doit()

    converter_adm_bssn_init_data = build_converter_adm_bssn_init_data(ccodes_dir, coord_system)
    converter_adm_bssn_init_data.doit()

    lapse_evolution_option = adapters.InterfaceParameter.build('BSSN.BSSN_gauge_RHSs::LapseEvolutionOption', numerical_integration.lapse_condition)
    shift_evolution_option = adapters.InterfaceParameter.build('BSSN.BSSN_gauge_RHSs::ShiftEvolutionOption', numerical_integration.shift_condition)

    betaU, enforce_detg_constraint_symb_expressions = build_bssn_rhs_symbolic_expressions(ccodes_dir, numerical_integration, dim)
    build_bssn_plus_scalarfield_rhss(ccodes_dir, betaU)
    build_ricci(ccodes_dir)
    build_hamiltonian(ccodes_dir)

    nrpy.EGC.output_Enforce_Detgammahat_Constraint_Ccode(ccodes_dir.root,exprs=enforce_detg_constraint_symb_expressions)

    build_cparamters_headers(ccodes_dir, coord_system)
    build_boundary_condition(ccodes_dir)
    build_scalar_field_collapse_playground_header(ccodes_dir, numerical_integration)
    build_scalar_field_collapse_playground_main(ccodes_dir)

    print("Now compiling, should take ~10 seconds...")
    mainc= ccodes_dir.make_under_root("ScalarFieldCollapse_Playground.c", is_dir=False)
    outc = ccodes_dir.make_under_outdir("ScalarFieldCollapse_Playground", is_dir=False)
    nrpy.cmd.C_compile(mainc, outc,compile_mode="optimized")
    print('End compile')

    os.chdir(ccodes_dir.outdir)
    nrpy.cmd.delete_existing_files("out*.txt")
    nrpy.cmd.delete_existing_files("out*.png")
    print(os.getcwd())
    print("Now running, should take ~20 seconds...")
    nrpy.cmd.Execute("ScalarFieldCollapse_Playground", "640 2 2 "+str(numerical_integration.cfl_factor),"out640.txt")
    print('finish running')
    os.chdir(os.path.join("../../"))

    visualize_all(ccodes_dir)
    visualize_conv(ccodes_dir, numerical_integration)

    return

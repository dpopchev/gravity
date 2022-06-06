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
from collections import namedtuple
from types import SimpleNamespace

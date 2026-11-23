"""Regression tests for the GCRODR adjoint solver (PETSc's HPDDM implementation of GCRO-DR).

GCRODR solves the same linear system as GMRES, so these tests check that it reproduces the GMRES adjoint and
sensitivities for every supported preconditioner type and preconditioner side. They require PETSc built with HPDDM.
"""

import copy
import os
import unittest
from itertools import product

import numpy as np
from baseclasses import AeroProblem
from mpi4py import MPI
from parameterized import parameterized_class

from adflow import ADFLOW
from reg_default_options import adflowDefOpts

baseDir = os.path.dirname(os.path.abspath(__file__))
gridFile = os.path.join(baseDir, "../../input_files/naca0012_rans-L2.cgns")

GLOBAL_PRECONDITIONERS = ["additive Schwarz", "multigrid"]
PRECONDITIONER_SIDES = ["right", "left"]
EVAL_FUNCS = ["cl", "cd", "cmz"]
AERO_DVS = ["alpha", "mach"]

# Both adjoints are converged to ADJOINT_L2_CONVERGENCE, so they should agree to roughly that level
ADJOINT_L2_CONVERGENCE = 1e-10
ADJOINT_REL_TOL = 1e-6
SENS_REL_TOL = 1e-6
SENS_ABS_TOL = 1e-10

commonTestOptions = {
    "gridFile": gridFile,
    "equationType": "RANS",
    "MGCycle": "sg",
    "L2Convergence": 1e-12,
    "nCycles": 2000,
    "adjointL2Convergence": ADJOINT_L2_CONVERGENCE,
    "adjointMaxIter": 1000,
    "adjointSubspaceSize": 100,
    "adjointAMGLevels": 2,
    "useBlockettes": False,
    "useANKSolver": True,
    "useNKSolver": True,
    "ANKSwitchTol": 10.0,
    "ANKSecondOrdSwitchTol": 1e-3,
    "ANKCoupledSwitchTol": 1e-6,
    "NKSwitchTol": 1e-8,
    "writeSurfaceSolution": False,
    "writeVolumeSolution": False,
}


def makeAeroProblem():
    """Create the NACA 0012 aero problem used by every test, with alpha and Mach as design variables."""
    ap = AeroProblem(
        name="naca0012",
        alpha=2.77,
        mach=0.6,
        reynolds=4800000.0,
        reynoldsLength=1.0,
        T=280.0,
        R=287.085,
        areaRef=1.0,
        chordRef=1.0,
        evalFuncs=EVAL_FUNCS,
    )
    for dv in AERO_DVS:
        ap.addDV(dv)
    return ap


def computeGlobalNorm(vec):
    """Compute the 2-norm of a vector that is distributed across all procs."""
    return np.sqrt(MPI.COMM_WORLD.allreduce(np.dot(vec, vec), op=MPI.SUM))


@parameterized_class(
    [
        {"globalPreconditioner": pc, "preconditionerSide": side}
        for pc, side in product(GLOBAL_PRECONDITIONERS, PRECONDITIONER_SIDES)
    ]
)
class TestGCRODRMatchesGMRES(unittest.TestCase):
    N_PROCS = 2

    # Set by parameterized_class
    globalPreconditioner = None
    preconditionerSide = None

    def solveFlowAndAdjoints(self, adjointSolver):
        """Solve the flow and compute the adjoint sensitivities of all EVAL_FUNCS with the given adjoint solver.

        Returns the aero problem, solver, and sensitivity dictionary.
        """
        options = copy.copy(adflowDefOpts)
        options["outputDirectory"] = os.path.join(baseDir, options["outputDirectory"])
        options.update(commonTestOptions)
        options.update(
            {
                "adjointSolver": adjointSolver,
                "globalPreconditioner": self.globalPreconditioner,
                "preconditionerSide": self.preconditionerSide,
            }
        )
        solver = ADFLOW(options=options, comm=MPI.COMM_WORLD, debug=False)
        ap = makeAeroProblem()
        solver(ap)
        funcsSens = {}
        solver.evalFunctionsSens(ap, funcsSens)
        return ap, solver, funcsSens

    def test_gcrodr_matches_gmres(self):
        """
        Given a converged RANS NACA 0012 flow and a choice of global preconditioner and preconditioner side,
        when the adjoints of several functions are solved in sequence with GCRODR and, separately, with GMRES,
        then GCRODR does not fail and its adjoint vectors and alpha and Mach sensitivities match the GMRES ones.
        """
        refAP, refSolver, refSens = self.solveFlowAndAdjoints("GMRES")
        ap, solver, sens = self.solveFlowAndAdjoints("GCRODR")

        self.assertFalse(refAP.adjointFailed)
        self.assertFalse(ap.adjointFailed)

        for func in EVAL_FUNCS:
            refAdjoint = refSolver.getAdjoint(func)
            adjoint = solver.getAdjoint(func)
            relError = computeGlobalNorm(adjoint - refAdjoint) / computeGlobalNorm(refAdjoint)
            self.assertLess(relError, ADJOINT_REL_TOL, msg=f"{func} adjoint")

            funcKey = f"{ap.name}_{func}"
            for dv in AERO_DVS:
                dvKey = f"{dv}_{ap.name}"
                np.testing.assert_allclose(
                    sens[funcKey][dvKey],
                    refSens[funcKey][dvKey],
                    rtol=SENS_REL_TOL,
                    atol=SENS_ABS_TOL,
                    err_msg=f"d{func}/d{dv}",
                )


if __name__ == "__main__":
    unittest.main()

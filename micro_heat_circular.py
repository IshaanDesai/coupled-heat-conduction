"""
Micro simulation
In this script we solve the Laplace equation with a grain depicted by a phase field on a square domain :math:`Ω`
with boundary :math:`Γ`, subject to periodic boundary conditions in both dimensions
"""
import math

from nutils import mesh, function, solver, export, sample, cli
import treelog
import numpy as np


class MicroSimulation:

    def __init__(self):
        """
        Constructor of MicroSimulation class.
        """
        # Initial parameters
        self._nelems = 20  # Elements in one direction
        self._ref_level = 3  # Number of levels of mesh refinement
        self._r_initial = 0.1  # Initial radius of the grain

        # Set up mesh with periodicity in both X and Y directions
        self._topo, self._geom = mesh.rectilinear([np.linspace(-0.5, 0.5, self._nelems)] * 2, periodic=(0, 1))
        self._topo_coarse = self._topo  # Save original coarse topology to use to re-refinement

        self._ns = None  # Namespace is created after initial refinement
        self._coarse_ns = None

        self._solu = None  # Solution of weights for which cell problem is solved for
        self._solphi = None  # Solution of phase field
        self._solphi_coarse = None  # Solution of phase field on original coarse topology
        self._solphinm1 = None  # Solution of phase field at t_{n-1}
        self._solphi_checkpoint = None  # Save the current solution of the phase field as a checkpoint.
        self._topo_checkpoint = None  # Save the refined mesh as a checkpoint.

        self._ucons = None

        self._first_iter_done = False
        self._initial_condition_is_set = False

    def initialize(self):
        # Define initial namespace
        self._ns = function.Namespace()
        self._ns.x = self._geom

        self._ns.phibasis = self._topo.basis('std', degree=1)
        self._ns.phi = 'phibasis_n ?solphi_n'  # Initial phase field
        self._ns.coarsephi = 'phibasis_n ?coarsesolphi_n'
        self._ns.lam = 2 / self._nelems  # Diffuse interface width is 2 cells on coarsest mesh

        # Initialize phase field
        solphi = self._get_analytical_phasefield(self._topo, self._ns, r=self._r_initial)

        # Refine the mesh
        self._topo, self._solphi = self._refine_mesh(self._topo, solphi)
        self._reinitialize_namespace(self._topo)
        self._initial_condition_is_set = True

        # Initialize phase field once more on refined topology
        solphi = self._get_analytical_phasefield(self._topo, self._ns, self._r_initial)

        target_porosity = 1 - math.pi * self._r_initial ** 2
        print("Target amount of void space = {}".format(target_porosity))

        # Solve phase field problem for a few steps to get the correct phase field
        psi = 0
        while psi < target_porosity:
            print("Solving Allen Cahn problem to achieve initial target grain structure")
            solphi = self.solve_allen_cahn(self._topo, solphi, 273, 0.001)
            psi = self.get_avg_porosity(solphi)

        # Save solution of phi
        self._solphi = solphi

        solu = self.solve_heat_cell_problem(solphi)
        b = self.get_eff_conductivity(solu, solphi)

        self.vtk_output('output-after-initialization', solu, solphi)

        output_data = dict()
        output_data["k_00"] = b[0][0]
        output_data["k_01"] = b[0][1]
        output_data["k_10"] = b[1][0]
        output_data["k_11"] = b[1][1]
        output_data["porosity"] = psi

        return output_data

    def _reinitialize_namespace(self, topo):
        self._ns = None  # Clear old namespace

        self._ns = function.Namespace()
        self._ns.x = self._geom
        self._ns.ubasis = topo.basis('h-std', degree=2).vector(topo.ndims)
        self._ns.phibasis = topo.basis('h-std', degree=1)
        self._ns.coarsephibasis = self._topo_coarse.basis('std', degree=1)

        # Physical constants
        self._ns.lam = 2 / (self._nelems * self._ref_level)  # Diffuse interface width
        self._ns.gam = 0.03
        self._ns.eqtemp = 273  # Equilibrium temperature
        self._ns.kg = 1.0  # Conductivity of grain material
        self._ns.ks = 5.0  # Conductivity of sand material
        self._ns.reacrate = '(?temp / eqtemp)^2 - 1'  # Constructed reaction rate based on macro temperature
        self._ns.u = 'ubasis_ni ?solu_n'  # Weights for which cell problem is solved for
        self._ns.du_ij = 'u_i,j'  # Gradient of weights field
        self._ns.phi = 'phibasis_n ?solphi_n'  # Phase field
        self._ns.projectedphi = 'phibasis_n ?projectedsolphi_n'  # Projected phase field
        self._ns.coarsephi = 'coarsephibasis_n ?coarsesolphi_n'  # Phase field on original coarse topology
        self._ns.ddwpdphi = '16 phi (1 - phi) (1 - 2 phi)'  # gradient of double-well potential
        self._ns.dphidt = 'phibasis_n (?solphi_n - ?solphinm1_n) / ?dt'  # Implicit time evolution of phase field

        self._ucons = np.zeros(len(self._ns.ubasis), dtype=bool)
        self._ucons[-1] = True  # constrain u to zero at a point

    @staticmethod
    def _analytical_phasefield(x, y, r, lam):
        return 1. / (1. + function.exp(-4. / lam * (function.sqrt(x ** 2 + y ** 2) - r + 0.001)))

    @staticmethod
    def _get_analytical_phasefield(topo, ns, r=0.25):
        phi_ini = MicroSimulation._analytical_phasefield(ns.x[0], ns.x[1], r, 0.066)
        sqrphi = topo.integral((ns.phi - phi_ini) ** 2, degree=2)
        solphi = solver.optimize('solphi', sqrphi, droptol=1E-12)

        return solphi

    def vtk_output(self, filename, solu, solphi):
        bezier = self._topo.sample('bezier', 2)
        x, u, phi = bezier.eval(['x_i', 'u_i', 'phi'] @ self._ns, solu=solu, solphi=solphi)
        with treelog.add(treelog.DataLog()):
            export.vtk(filename, bezier.tri, x, T=u, phi=phi)

    def save_checkpoint(self):
        self._solphi_checkpoint = self._solphi
        self._topo_checkpoint = self._topo

    def revert_checkpoint(self):
        self._solphi = self._solphi_checkpoint
        self._topo = self._topo_checkpoint

    def _refine_mesh(self, topo_nm1, solphi_nm1):
        """
        At the time of the calling of this function a predicted solution exists in ns.phi
        """
        if not self._initial_condition_is_set:
            solphi = solphi_nm1
            # ----- Refine the coarse mesh according to the projected solution to get a predicted refined topology ----
            topo = self._topo_coarse
            for level in range(self._ref_level):
                print("refinement level = {}".format(level))
                smpl = topo.sample('uniform', 5)
                ielem, criterion = smpl.eval([topo.f_index, abs(self._ns.coarsephi - .5) < .4],
                                             coarsesolphi=solphi)

                # Refine the elements for which at least one point tests true.
                topo = topo.refined_by(np.unique(ielem[criterion]))

                self._reinitialize_namespace(topo)

                phi_ini = MicroSimulation._analytical_phasefield(self._ns.x[0], self._ns.x[1], self._r_initial,
                                                                 self._ns.lam)
                sqrphi = topo.integral((self._ns.coarsephi - phi_ini) ** 2, degree=2)
                solphi = solver.optimize('coarsesolphi', sqrphi, droptol=1E-12)
            # ----------------------------------------------------------------------------------------------------
        else:
            # ----- Refine the coarse mesh according to the projected solution to get a predicted refined topology ----
            topo = self._topo_coarse
            for level in range(self._ref_level):
                print("refinement level = {}".format(level))
                topo_union1 = topo_nm1 & topo
                smpl = topo_union1.sample('uniform', 5)
                ielem, criterion = smpl.eval([topo.f_index, abs(self._ns.phi - .5) < .4],
                                             solphi=solphi_nm1)

                # Refine the elements for which at least one point tests true.
                topo = topo.refined_by(np.unique(ielem[criterion]))
            # ----------------------------------------------------------------------------------------------------

            # Create a new projection mesh which is the union of the previous refined mesh and the predicted mesh
            topo_union = topo_nm1 & topo

            # ----- Project the solution of the last time step on the projection mesh -----
            self._ns.projectedphi = function.dotarg('projectedsolphi', topo.basis('h-std', degree=1))
            sqrphi = topo_union.integral((self._ns.projectedphi - self._ns.phi) ** 2, degree=2)
            solphi = solver.optimize('projectedsolphi', sqrphi, droptol=1E-12, arguments=dict(solphi=solphi_nm1))

        return topo, solphi

    def solve_allen_cahn(self, topo, phi_coeffs_nm1, temperature, dt):
        """
        Solving the Allen-Cahn Equation using a Newton solver.
        Returns porosity of the micro domain
        """
        self._first_iter_done = True
        resphi = topo.integral('(lam^2 phibasis_n dphidt + gam phibasis_n ddwpdphi + gam lam^2 phibasis_n,i phi_,i + '
                               '4 lam reacrate phibasis_n phi (1 - phi)) d:x' @ self._ns, degree=2)

        args = dict(solphinm1=phi_coeffs_nm1, dt=dt, temp=temperature)
        phi_coeffs = solver.newton('solphi', resphi, lhs0=phi_coeffs_nm1, arguments=args).solve(tol=1e-10)

        return phi_coeffs

    def get_avg_porosity(self, phi_coeffs):
        psi = self._topo.integral('phi d:x' @ self._ns, degree=2).eval(solphi=phi_coeffs)

        return psi

    def solve_heat_cell_problem(self, phi_coeffs):
        """
        Solving the P1 homogenized heat equation
        Returns upscaled conductivity matrix for the micro domain
        """
        res = self._topo.integral('((phi ks + (1 - phi) kg) u_i,j ubasis_ni,j + '
                                  'ubasis_ni,j (phi ks + (1 - phi) kg) $_ij) d:x' @ self._ns, degree=4)

        args = dict(solphi=phi_coeffs)
        u_coeffs = solver.solve_linear('solu', res, constrain=self._ucons, arguments=args)

        return u_coeffs

    def get_eff_conductivity(self, u_coeffs, phi_coeffs):
        b = self._topo.integral(self._ns.eval_ij('(phi ks + (1 - phi) kg) ($_ij + du_ij) d:x'), degree=4).eval(
            solu=u_coeffs, solphi=phi_coeffs)

        return b.export("dense")

    def solve(self, macro_data, dt):
        self._topo, self._solphi = self._refine_mesh(self._topo, self._solphi)
        self._reinitialize_namespace(self._topo)

        self._solphi = self.solve_allen_cahn(self._topo, self._solphi, macro_data["temperature"], dt)
        psi = self.get_avg_porosity(self._solphi)
        print("Upscaled relative amount of sand material = {}".format(psi))

        solu = self.solve_heat_cell_problem(self._solphi)
        b = self.get_eff_conductivity(solu, self._solphi)
        print("Upscaled conductivity = {}".format(b))

        output_data = dict()
        output_data["k_00"] = b[0][0]
        output_data["k_01"] = b[0][1]
        output_data["k_10"] = b[1][0]
        output_data["k_11"] = b[1][1]
        output_data["porosity"] = psi

        return output_data


def main():
    micro_problem = MicroSimulation()
    dt = 1e-3
    micro_problem.initialize()
    temp_values = np.arange(273.0, 583.0, 20.0)
    t = 0.0

    for temperature in temp_values:
        print("t = {}".format(t))
        micro_sim_output = micro_problem.solve(temperature, dt)
        print(micro_sim_output)
        t += dt


if __name__ == "__main__":
    cli.run(main)

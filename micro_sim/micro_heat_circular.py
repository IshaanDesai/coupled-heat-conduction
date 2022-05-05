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
        # Elements in one direction
        self._nelems = 20

        # Number of levels of mesh refinement
        self._ref_level = 3

        # Set up mesh with periodicity in both X and Y directions
        self._topo, self._geom = mesh.rectilinear([np.linspace(-0.5, 0.5, self._nelems)] * 2, periodic=(0, 1))
        self._topo_coarse = self._topo  # Save original coarse topology to use to re-refinement

        self._ns = None  # Namespace is created after initial refinement
        self._coarse_ns = None

        self._solu = None  # Solution of weights for which cell problem is solved for
        self._solphi = None  # Solution of phase field
        self._solphi_coarse = None  # Solution of phase field on original coarse topology
        self._solphinm1 = None  # Solution of phase field at t_{n-1}
        self._solphi_checkpoint = None  # Checkpointing state of phase field. Defined in first save of state

        self._ucons = None

        self._first_iter_done = False
        self._initial_phasefield_setting = False

    def initialize(self, dt):
        # Define initial namespace
        self._ns = function.Namespace()
        self._ns.x = self._geom

        # Initial state of phase field
        self._ns.phibasis = self._topo.basis('std', degree=1)
        self._ns.phi = 'phibasis_n ?solphi_n'  # Initial phase field
        self._ns.coarsephi = 'phibasis_n ?coarsesolphi_n'
        self._ns.lam = 4 / self._nelems  # Diffuse interface width is 4 cells

        # Initialize phase field
        self._solphi = self._get_analytical_phasefield(self._topo, self._ns)

        # Refine the mesh
        self.refine_mesh()
        self._reinitialize_namespace(self._topo)
        self._initial_phasefield_setting = True

        # Initialize phase field once more on refined topology
        self._solphi = self._get_analytical_phasefield(self._topo, self._ns)

        self._solphinm1 = self._solphi  # At t = 0 the history data is same as the new data

        r = 0.25
        target_poro = 1 - math.pi * r ** 2
        print("Target amount of void space = {}".format(target_poro))

        # Solve phase field problem for a few steps to get the correct phase field
        poro = 0
        while poro < target_poro:
            print("Solving Allen Cahn problem to achieve initial target grain structure")
            poro = self.solve_allen_cahn(273, dt)

        b = self.solve_heat_cell_problem()

        return b, poro

    def _reinitialize_namespace(self, topo):
        self._ns = None  # Clear old namespace

        self._ns = function.Namespace()
        self._ns.x = self._geom
        self._ns.ubasis = topo.basis('h-std', degree=2).vector(topo.ndims)
        self._ns.phibasis = topo.basis('h-std', degree=1)
        self._ns.coarsephibasis = self._topo_coarse.basis('std', degree=1)

        # Physical constants
        self._ns.lam = 4 / (self._nelems * self._ref_level)  # Diffuse interface width is 4 cells on finest refinement
        self._ns.gam = 0.03
        self._ns.eqtemp = 273  # Equilibrium temperature
        self._ns.kg = 1.0  # Conductivity of grain material
        self._ns.ks = 5.0  # Conductivity of sand material

        self._ns.reacrate = '(?temp / eqtemp)^2 - 1'  # Constructed reaction rate based on macro temperature
        self._ns.u = 'ubasis_ni ?solu_n'  # Weights for which cell problem is solved for
        self._ns.du_ij = 'u_i,j'  # Gradient of weights field
        self._ns.phi = 'phibasis_n ?solphi_n'  # Phase field
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

    def vtk_output(self):
        bezier = self._topo.sample('bezier', 2)
        x, u, phi = bezier.eval(['x_i', 'u_i', 'phi'] @ self._ns, solu=self._solu, solphi=self._solphi)
        with treelog.add(treelog.DataLog()):
            export.vtk('micro-heat', bezier.tri, x, T=u, phi=phi)

    def save_state(self):
        self._solphi_checkpoint = self._solphinm1

    def revert_state(self):
        self._solphinm1 = self._solphi_checkpoint

    def refine_mesh(self):
        """
        At the time of the calling of this function a predicted solution exists in ns.phi
        """
        if self._first_iter_done:
            print("Performing the coarsening step")
            # Project the current auxiliary solution onto coarse mesh
            self._ns.coarsephi = function.dotarg('coarsesolphi', self._topo_coarse.basis('std', degree=1))
            sqrphi = self._topo.integral((self._ns.coarsephi - self._ns.phi) ** 2, degree=2)

            print("Performing the projection step")
            argument = dict(solphi=self._solphi)
            solphi_projected = solver.optimize('coarsesolphi', sqrphi, droptol=1E-12, arguments=argument)
        else:
            print("First step, no coarsening required")
            solphi_projected = self._solphi

        if not self._initial_phasefield_setting:
            # Refine the coarse mesh according to the predicted solution to get a predicted refined topology
            topo = self._topo_coarse  # Set the predicted topology as the initial coarse topology
            for level in range(self._ref_level):
                print("refinement level = {}".format(level))
                smpl = topo.sample('uniform', 5)
                ielem, criterion = smpl.eval([topo.f_index, abs(self._ns.coarsephi - .5) < .4],
                                             coarsesolphi=solphi_projected)

                # Refine the elements for which at least one point tests true.
                topo = topo.refined_by(np.unique(ielem[criterion]))
                self._reinitialize_namespace(topo)
                phi_ini = MicroSimulation._analytical_phasefield(self._ns.x[0], self._ns.x[1], 0.25, 0.066)
                sqrphi = topo.integral((self._ns.coarsephi - phi_ini) ** 2, degree=2)
                solphi_projected = solver.optimize('coarsesolphi', sqrphi, droptol=1E-12)
        else:
            # Refine the coarse mesh according to the predicted solution to get a predicted refined topology
            topo = self._topo_coarse  # Set the predicted topology as the initial coarse topology
            for level in range(self._ref_level):
                print("refinement level = {}".format(level))
                smpl = topo.sample('uniform', 5)
                ielem, criterion = smpl.eval([topo.f_index, abs(self._ns.coarsephi - .5) < .4],
                                             coarsesolphi=solphi_projected)

                # Refine the elements for which at least one point tests true.
                topo = topo.refined_by(np.unique(ielem[criterion]))

        # Create a projection topology which is the union of refined topologies of previous time step and the predicted
        self._topo = self._topo & topo

    def solve_allen_cahn(self, temperature, dt):
        """
        Solving the Allen-Cahn Equation using a Newton solver.
        Returns porosity of the micro domain
        """
        self._first_iter_done = True
        resphi = self._topo.integral(
            '(lam^2 phibasis_n dphidt + gam phibasis_n ddwpdphi + gam lam^2 phibasis_n,i phi_,i + '
            '4 lam reacrate phibasis_n phi (1 - phi)) d:x' @ self._ns, degree=2)

        args = dict(solphinm1=self._solphinm1, dt=dt, temp=temperature)
        self._solphi = solver.newton('solphi', resphi, lhs0=self._solphinm1, arguments=args).solve(tol=1e-10)

        # Update the state
        self._solphinm1 = self._solphi

        # Calculating ratio of grain amount for upscaling
        psi = self._topo.integral('phi d:x' @ self._ns, degree=2).eval(solphi=self._solphi)
        print("Upscaled relative amount of sand material = {}".format(psi))

        return psi

    def solve_heat_cell_problem(self):
        """
        Solving the P1 homogenized heat equation
        Returns upscaled conductivity matrix for the micro domain
        """
        res = self._topo.integral('((phi ks + (1 - phi) kg) u_i,j ubasis_ni,j + '
                                  'ubasis_ni,j (phi ks + (1 - phi) kg) $_ij) d:x' @ self._ns, degree=4)

        args = dict(solphi=self._solphi)
        self._solu = solver.solve_linear('solu', res, constrain=self._ucons, arguments=args)

        # Calculating effective conductivity for upscaling
        b = self._topo.integral(self._ns.eval_ij('(phi ks + (1 - phi) kg) ($_ij + du_ij) d:x'), degree=4).eval(
            solu=self._solu, solphi=self._solphi)

        print("Upscaled conductivity = {}".format(b.export("dense")))

        return b.export("dense")


def main():
    micro_problem = MicroSimulation()
    dt = 1e-3
    micro_problem.initialize(dt)
    micro_problem.vtk_output()

    temp_values = np.arange(273.0, 283.0, 10.0)
    t = 0.0

    for temperature in temp_values:
        print("t = {}".format(t))
        micro_problem.refine_mesh()
        micro_problem.solve_allen_cahn(temperature, dt)
        micro_problem.solve_heat_cell_problem()
        micro_problem.vtk_output()
        t += dt


if __name__ == "__main__":
    cli.run(main)

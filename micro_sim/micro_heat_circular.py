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
        self._ref_level = 4

        # Set up mesh with periodicity in both X and Y directions
        self._topo, self._geom = mesh.rectilinear([np.linspace(-0.5, 0.5, self._nelems)] * 2, periodic=(0, 1))
        self._topo_coarse = self._topo  # Save original coarse topology to use to re-refinement

        self._ns = None  # Namespace is created after initial refinement

        self._solu = None  # Solution of weights for which cell problem is solved for
        self._solphi = None  # Solution of phase field
        self._solphinm1 = None  # Solution of phase field at t_{n-1}
        self._solphi_checkpoint = None  # Checkpointing state of phase field. Defined in first save of state

        self._ucons = None

        self._first_iter_done = False

    def initialize(self, dt):
        r = 0.25  # initial grain radius

        # Define initial namespace
        self._ns = function.Namespace()
        self._ns.x = self._geom

        # Initial state of phase field
        self._ns.phibasis = self._topo.basis('std', degree=1)
        self._ns.phi = 'phibasis_n ?solphi_n'  # Initial phase field

        # Initialize phase field
        self.initialize_phasefield(r)

        # Refine the mesh
        self.refine_mesh()

        # Initialize phase field once more on refined topology
        self.initialize_phasefield(r)

        self._solphinm1 = self._solphi  # At t = 0 the history data is same as the new data

        target_poro = 1 - math.pi * r**2
        print("Target amount of void space = {}".format(target_poro))

        # Solve phase field problem for a few steps to get the correct phase field
        poro = 0
        while poro < target_poro:
            poro = self.solve_allen_cahn(273, dt)

        b = self.solve_heat_cell_problem()

        return b, poro

    def reinitialize_namespace(self):
        self._ns = None  # Clear old namespace

        self._ns = function.Namespace()
        self._ns.x = self._geom
        self._ns.ubasis = self._topo.basis('h-std', degree=2).vector(self._topo.ndims)
        self._ns.phibasis = self._topo.basis('h-std', degree=1)

        # Physical constants
        self._ns.lam = 3 / (self._nelems * self._ref_level)  # Diffuse interface width is 4 cells on finest refinement
        self._ns.gam = 0.03
        self._ns.eqtemp = 273  # Equilibrium temperature
        self._ns.kg = 1.0  # Conductivity of grain material
        self._ns.ks = 5.0  # Conductivity of sand material

        self._ns.reacrate = '(?temp / eqtemp)^2 - 1'  # Constructed reaction rate based on macro temperature
        self._ns.u = 'ubasis_ni ?solu_n'  # Weights for which cell problem is solved for
        self._ns.du_ij = 'u_i,j'  # Gradient of weights field
        self._ns.phi = 'phibasis_n ?solphi_n'  # Phase field
        self._ns.ddwpdphi = '16 phi (1 - phi) (1 - 2 phi)'  # gradient of double-well potential
        self._ns.dphidt = 'phibasis_n (?solphi_n - ?solphinm1_n) / ?dt'  # Implicit time evolution of phase field

        self._ucons = np.zeros(len(self._ns.ubasis), dtype=bool)
        self._ucons[-1] = True  # constrain u to zero at a point

    def initialize_phasefield(self, r=0.25):
        phi_ini = self._initial_phasefield(self._ns.x[0], self._ns.x[1], r, 0.066)
        sqrphi = self._topo.integral((self._ns.phi - phi_ini) ** 2, degree=2)
        self._solphi = solver.optimize('solphi', sqrphi, droptol=1E-12)

    def _initial_phasefield(self, x, y, r, lam):
        return 1. / (1. + function.exp(-4. / lam * (function.sqrt(x ** 2 + y ** 2) - r + 0.001)))

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
        topo_refined = None

        if self._first_iter_done:
            print("Performing the coarsening step")
            # Project the current auxiliary solution onto coarse mesh
            coarse_solphi = function.dotarg('solphi', self._topo_coarse.basis('std', degree=1))
            print("Shape of coarse_solphi = {}, shape of ns.phi = {}".format(coarse_solphi.shape, self._ns.phi.shape))
            sqrphi = self._topo.integral((self._ns.phi - coarse_solphi) ** 2, degree=2)
            solphi = solver.optimize('solphi', sqrphi, droptol=1E-12)
        else:
            print("First step, no coarsening required")
            solphi = self._solphi

        print("Performing the refining step")
        # Refine the coarse mesh according to the predicted solution to get a predicted refined topology
        topo_predicted = self._topo_coarse  # Set the predicted topology as the initial coarse topology
        for level in range(self._ref_level):
            print("level = {}".format(level))
            smpl = self._topo_coarse.sample('uniform', 5)
            ielem, criterion = smpl.eval([topo_predicted.f_index, abs(self._ns.phi - .5) < .4], solphi=solphi)

            # Refine the elements for which at least one point tests true.
            topo_refined = topo_predicted.refined_by(np.unique(ielem[criterion]))

        # Create a projection topology which is the union of refined topologies of previous time step and the predicted
        self._topo = self._topo & topo_refined

        # Reinitialize the namespace according to the refined topology
        self.reinitialize_namespace()

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

    temp_values = np.arange(273.0, 493.0, 20.0)
    t = 0.0

    for temperature in temp_values:
        micro_problem.refine_mesh()
        micro_problem.solve_allen_cahn(temperature, dt)
        micro_problem.solve_heat_cell_problem()
        micro_problem.vtk_output()
        t += dt
        print("time t = {}".format(t))


if __name__ == "__main__":
    cli.run(main)

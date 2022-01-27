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
        self._topo_old = self._topo  # Save original coarse topology to use to re-refinement

        self._ns = None  # Namespace is created after initial refinement

        self._solu = None  # Solution of weights for which cell problem is solved for
        self._solphi = None  # Solution of phase field
        self._solphinm1 = None  # Solution of phase field at t_{n-1}
        self._solphi_checkpoint = None  # Checkpointing state of phase field. Defined in first save of state

        self._ucons = None

    def initialize(self, dt):
        r = 0.25  # initial grain radius

        # Define initial namespace
        nsi = function.Namespace()
        nsi.x = self._geom

        # Initial state of phase field
        nsi.phibasis = self._topo.basis('std', degree=1)
        nsi.phi = 'phibasis_n ?solphi_n'  # Initial phase field
        phi_ini = self._initial_phasefield(nsi.x[0], nsi.x[1], r, 0.066)
        sqrphi = self._topo.integral((nsi.phi - phi_ini) ** 2, degree=2)
        self._solphi = solver.optimize('solphi', sqrphi, droptol=1E-12)

        # Refine the mesh
        self.refine_mesh(nsi)

        nsi = None  # Delete initial namespace

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

        phi_ini = self._initial_phasefield(self._ns.x[0], self._ns.x[1], r, 0.066)
        sqrphi = self._topo.integral((self._ns.phi - phi_ini) ** 2, degree=2)
        self._solphi = solver.optimize('solphi', sqrphi, droptol=1E-12)

        self._ucons = np.zeros(len(self._ns.ubasis), dtype=bool)
        self._ucons[-1] = True  # constrain u to zero at a point

        self._solphinm1 = self._solphi  # At t = 0 the history data is same as the new data

        target_poro = 1 - math.pi * r**2
        print("Target amount of sand material = {}".format(target_poro))

        # Solve phase field problem for a few steps to get the correct phase field
        poro = 0
        while poro < target_poro:
            poro = self.solve_allen_cahn(273, dt)

        b = self.solve_heat_cell_problem()

        return b, poro

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

    def refine_mesh(self, ns):
        for level in range(self._ref_level):
            print("level = {}".format(level))
            smpl = self._topo.sample('uniform', 5)
            ielem, criterion = smpl.eval([self._topo.f_index, abs(ns.phi - .5) < .4], solphi=self._solphi)

            # Refine the elements for which at least one point tests true.
            self._topo = self._topo.refined_by(np.unique(ielem[criterion]))

    def solve_allen_cahn(self, temperature, dt):
        """
        Solving the Allen-Cahn Equation using a Newton solver.
        Returns upscaled ratio of grain and surrounding material
        """
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

    #temp_values = np.arange(273.0, 350.0, 1.0)
    #t = 0.0

    #for temperature in temp_values:
    #    micro_problem.solve_allen_cahn(temperature, dt)
    #    micro_problem.solve_heat_cell_problem()
    #    micro_problem.vtk_output()
    #    t += dt
    #    print("time t = {}".format(t))


if __name__ == "__main__":
    cli.run(main)

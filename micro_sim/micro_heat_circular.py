"""
Micro simulation
In this script we solve the Laplace equation with a grain depicted by a phase field on a square domain :math:`Ω`
with boundary :math:`Γ`, subject to periodic boundary conditions in both dimensions
"""
import math

from nutils import mesh, function, solver, export, sample
import treelog
import numpy as np


class MicroSimulation:

    def __init__(self):
        """
        Constructor of MicroSimulation class.
        """
        # Elements in one direction
        nelems = 40

        # Set up mesh with periodicity in both X and Y directions
        self._topo, self._geom = mesh.rectilinear([np.linspace(-0.5, 0.5, nelems)] * 2, periodic=(0, 1))
        self._topo_ref = self._topo  # refined topology which is initialized in solve()

        self._ns = function.Namespace()
        self._ns.x = self._geom
        self._ns.ubasis = self._topo.basis('std', degree=2).vector(self._topo.ndims)
        self._ns.phibasis = self._topo.basis('std', degree=1)

        # Physical constants
        self._ns.lam = 0.02
        self._ns.gam = 0.03
        self._ns.eqtemp = 273  # Equilibrium temperature
        self._ns.kg = 5.0  # Conductivity of grain material
        self._ns.ks = 1.0  # Conductivity of sand material

        self._ns.reacrate = '(?temp / eqtemp)^2 - 1'
        self._ns.u = 'ubasis_ni ?solu_n'
        self._ns.du_ij = 'u_i,j'
        self._ns.phi = 'phibasis_n ?solphi_n'
        self._ns.ddwpdphi = '16 phi (1 - phi) (1 - 2 phi)'  # gradient of double-well potential
        self._ns.dphidt = 'lam^2 phibasis_n (?solphi_n - ?solphi0_n) / ?dt'

        self._solu = None  # Solution of u
        self._solphi = None  # Solution of phi
        self._solphi_checkpoint = None  # Defined in first save of state

        self._r = 0.25  # grain radius of current time step (set initial value at this point)

        self._ucons = np.zeros(len(self._ns.ubasis), dtype=bool)
        self._ucons[-1] = True  # constrain u to zero at a point

        # Initial state of phase field: circular grain with radius 0.25
        phi_ini = self._initial_phasefield(self._ns.x[0], self._ns.x[1], 0.25)
        sqrphi = self._topo.integral((self._ns.phi - phi_ini) ** 2, degree=2)
        self._solphi0 = solver.optimize('solphi', sqrphi, droptol=1E-12)

    def initialize(self, dt):
        # Solve phase field problem for a few steps to get the correct phase field
        target_poro = 1 - math.pi * self._r ** 2
        print("Intial target porosity = {}".format(target_poro))

        poro = 0
        while poro < target_poro:
            poro = self.solve_allen_cahn(273, dt)

        b = self.solve_heat_cell_problem()

        return b, poro

    def _initial_phasefield(self, x, y, r):
        return 1. / (1. + function.exp(-4. / self._ns.lam * (function.sqrt(x ** 2 + y ** 2) - r + 0.001)))

    def vtk_output(self, rank):
        bezier = self._topo_ref.sample('bezier', 2)
        x, u, phi = bezier.eval(['x_i', 'u_i', 'phi'] @ self._ns, solu=self._solu, solphi=self._solphi)
        with treelog.add(treelog.DataLog()):
            export.vtk('micro-heat-' + str(rank), bezier.tri, x, T=u, phi=phi)

    def save_state(self):
        self._solphi_checkpoint = self._solphi0

    def revert_state(self):
        self._solphi0 = self._solphi_checkpoint

    def refine_mesh(self):
        self._topo_ref = self._topo
        # dist = abs(self._r - function.norm2(self._geom))
        # for margin in self._r / 2, self._r / 4, self._r / 8:
        #     # refine elements within `margin` of the circle boundary
        #     active, ielem = self._topo_ref.sample('bezier', 2).eval([margin - dist, self._topo_ref.f_index])
        #     self._topo_ref = self._topo_ref.refined_by(np.unique(ielem[active > 0]))

    def solve_allen_cahn(self, temperature, dt):
        resphi = self._topo_ref.integral(
            '(lam^2 phibasis_n dphidt + gam phibasis_n ddwpdphi + gam lam^2 phibasis_n,i phi_,i + '
            '4 lam reacrate phibasis_n phi (1 - phi)) d:x' @ self._ns,
            degree=2)

        args = dict(solphi0=self._solphi0, dt=dt, temp=temperature)
        self._solphi = solver.newton('solphi', resphi, lhs0=self._solphi0, arguments=args).solve(tol=1e-10)

        # Update the state
        self._solphi0 = self._solphi

        # Calculating porosity for upscaling
        psi = self._topo_ref.integral('phi d:x' @ self._ns, degree=2).eval(solphi=self._solphi)
        print("Upscaled porosity = {}".format(psi))

        return psi

    def solve_heat_cell_problem(self):
        res = self._topo_ref.integral('(phi ks + (1 - phi) kg) u_i,j ubasis_ni,j d:x' @ self._ns, degree=4)
        res += self._topo_ref.integral('ubasis_ni,j (phi ks + (1 - phi) kg) $_ij d:x' @ self._ns, degree=4)

        args = dict(solphi=self._solphi)
        self._solu = solver.solve_linear('solu', res, constrain=self._ucons, arguments=args)

        # Calculating effective conductivity for upscaling
        b = self._topo_ref.integral(self._ns.eval_ij('(phi ks + (1 - phi) kg) ($_ij + du_ij) d:x'), degree=4).eval(
            solu=self._solu, solphi=self._solphi)

        print("Upscaled conductivity = {}".format(b.export("dense")))

        return b.export("dense")

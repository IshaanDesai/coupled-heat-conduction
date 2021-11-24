"""
Micro simulation
In this script we solve the Laplace equation with a grain depicted by a phase field on a square domain :math:`Ω`
with boundary :math:`Γ`, subject to periodic boundary conditions in both dimensions
"""


from nutils import mesh, function, solver, export, sample
import treelog
import numpy as np


class MicroSimulation:

    def __init__(self):
        """
        Constructor of MicroSimulation class.
        """
        # Constants
        self._temperature_eq = 273

        # Elements in one direction
        nelems = 10

        # Set up mesh with periodicity in both X and Y directions
        self._topo, self._geom = mesh.rectilinear([np.linspace(-0.5, 0.5, nelems)] * 2, periodic=(0, 1))
        self._topo_ref = None  # refined topology which is initialized in solve()

        self._ns = function.Namespace()
        self._ns.x = self._geom
        self._ns.ubasis = self._topo.basis('std', degree=2).vector(self._topo.ndims)
        self._ns.phibasis = self._topo.basis('std', degree=1)

        # Physical variables
        self._ns.dt = 0.1  # Initial time step guess
        self._ns.lam = 0.05
        self._ns.gam = 1
        self._ns.reacrate = 1.0

        self._ns.u = 'ubasis_ni ?solu_n'

        self._ns.du_ij = 'u_i,j'
        self._ns.phi = 'phibasis_n ?solphi_n'

        self._ns.ddwpdphi = '16 phi ( 1 - phi ) ( 1 - 2 phi )'  # gradient of double-well potential
        self._ns.dphidt = 'lam^2 phibasis_n (?solphi_n - ?solphi0_n) / dt'

        # Conductivity of grain material
        self._ns.kg = 5.0
        # Conductivity of sand material
        self._ns.ks = 1.0

        # Solution of u
        self._solu = None

        # Solution of phi
        self._solphi = None
        self._solphi_checkpoint = None  # Defined in first save of state

        # Radius from previous time step
        self._r = 0.25  # grain radius of current time step (set initial value at this point)
        self._r_cp = 0  # grain radius value used for checkpointing

        self._ucons = np.zeros(len(self._ns.ubasis), dtype=bool)
        self._ucons[-1] = True  # constrain u to zero at a point

        # Initial state of phase field: circular grain with radius 0.25
        self._ns.phiini = self._initial_phasefield(self._ns.x[0], self._ns.x[1], 0.25)
        sqrphi = self._topo.integral('(phi - phiini)^2' @ self._ns, degree=1)
        self._solphi0 = solver.optimize('solphi', sqrphi, droptol=1E-12)

    def _initial_phasefield(self, x, y, r):
        return 1. / (1. + function.exp(-4. / self._ns.lam * (function.sqrt(x ** 2 + y ** 2) - r)))

    def vtk_output(self, rank):
        bezier = self._topo_ref.sample('bezier', 2)
        x, u, phi = bezier.eval(['x_i', 'u_i', 'phi'] @ self._ns, solu=self._solu)
        with treelog.add(treelog.DataLog()):
            export.vtk('micro-heat-' + str(rank), bezier.tri, x, T=u, phi=phi)

    def initialize(self, temperature=273, dt=0.001):
        b, psi = self.solve(temperature, dt)

        return b, psi

    def save_state(self):
        self._solphi_checkpoint = self._solphi0

    def revert_state(self):
        self._solphi0 = self._solphi_checkpoint

    def solve(self, temperature, dt):
        """
        Function which solves the steady state cell problem to calculate weights which are solutions to P1 problem
        of homogenized
        """
        self._ns.reacrate = (temperature ** 2 / self._temperature_eq ** 2) - 1
        self._ns.dt = dt

        self._topo_ref = self._topo
        # dist = abs(self._r - function.norm2(self._geom))
        # for margin in self._r / 2, self._r / 4, self._r / 8:
        #     # refine elements within `margin` of the circle boundary
        #     active, ielem = self._topo_ref.sample('bezier', 2).eval([margin - dist, self._topo_ref.f_index])
        #     self._topo_ref = self._topo_ref.refined_by(np.unique(ielem[active > 0]))

        ############################
        # Phase field problem      #
        ############################

        resphi = self._topo_ref.integral(
            '(lam^2 phibasis_n dphidt + phibasis_n ddwpdphi + gam lam^2 phibasis_n,i phi_,i) d:x' @ self._ns,
            degree=2)
        resphi += self._topo_ref.integral('(4 lam reacrate phibasis_n phi (1 - phi)) d:x' @ self._ns, degree=2)

        self._solphi = solver.newton('solphi', resphi, lhs0=self._solphi0)

        #############################
        # Steady-state cell problem #
        #############################

        res = self._topo_ref.integral('(phi ks + (1 - phi) kg) u_i,j ubasis_ni,j d:x' @ self._ns, degree=4)
        res += self._topo_ref.integral('ubasis_ni,j (phi ks + (1 - phi) kg) $_ij d:x' @ self._ns, degree=4)

        self._solu = solver.solve_linear('solu', res, constrain=self._ucons, arguments=dict(solphi=self._solphi))

        # upscaling
        b = self._topo_ref.integral(self._ns.eval_ij('(phi ks + (1 - phi) kg) ($_ij + du_ij) d:x'), degree=4).eval(
            solu=self._solu)
        psi = self._topo_ref.integral('phi d:x' @ self._ns, degree=2).eval(solu=self._solu)

        # Update the state
        self._solphi0 = self._solphi

        # print("Upscaled conductivity = {}".format(b.export("dense")))
        # print("Upscaled porosity = {}".format(psi))

        return b.export("dense"), psi

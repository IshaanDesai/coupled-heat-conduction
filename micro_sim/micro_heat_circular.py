"""
Micro simulation
In this script we solve the Laplace equation with a grain depicted by a phase field on a square domain :math:`Ω`
with boundary :math:`Γ`, subject to periodic boundary conditions in both dimensions
"""


from nutils import mesh, function, solver, export, cli
import treelog
import numpy as np


class MicroSimulation:

    def __init__(self):
        """
        Constructor of MicroSimulation class.
        """
        # Elements in one direction
        nelems = 10

        # Set up mesh with periodicity in both X and Y directions
        self._domain, geom = mesh.rectilinear([np.linspace(-0.5, 0.5, nelems)] * 2, periodic=(0, 1))

        self._ns = function.Namespace()
        self._ns.x = geom
        self._ns.basis = self._domain.basis('std', degree=2).vector(2)
        self._ns.u = 'basis_ni ?solu_n'
        self._ns.du_ij = 'u_i,j'

        # Conductivity of grain material
        self._ns.kg = 5.0
        # Conductivity of sand material
        self._ns.ks = 1.0

        # Solution of u
        self._solu = None

        # Radius from previous time step
        self._r_nm1 = 0.1

        # Prepare the post processing sample
        self._bezier = self._domain.sample('bezier', 2)

        self._ucons = np.zeros(len(self._ns.basis), dtype=bool)
        self._ucons[-1] = True  # constrain u to zero at a point

    def _update_radius(self, r, temperature, dt):
        temperature_eq = 273

        return r + dt * ((temperature ** 2 / temperature_eq ** 2) - 1)

    def _phasefield(self, x, y, r):
        lam = 0.1

        phi = 1. / (1. + function.exp(-4. / lam * (function.sqrt(x ** 2 + y ** 2) - r)))

        return phi

    def vtk_output(self):
        # Output phase field
        x, phi = self._bezier.eval(['x_i', 'phi'] @ self._ns)
        with treelog.add(treelog.DataLog()):
            export.vtk('phase-field', self._bezier.tri, x, phi=phi)

        # u value
        x, u = self._bezier.eval(['x_i', 'u_i'] @ self._ns, solu=self._solu)
        with treelog.add(treelog.DataLog()):
            export.vtk('u-value', self._bezier.tri, x, T=u)

    def initialize(self, temperature=273, dt=0.001):
        b, psi = self.solve(temperature, dt)

        return b, psi

    def solve(self, temperature, dt):
        """
        TODO Description
        """
        r = self._update_radius(temperature, self._r_nm1, dt)
        self._ns.phi = self._phasefield(self._ns.x[0], self._ns.x[1], r)

        # Define cell problem
        res = self._domain.integral('(phi ks + (1 - phi) kg) u_i,j basis_ni,j d:x' @ self._ns, degree=4)
        res += self._domain.integral('basis_ni,j (phi ks + (1 - phi) kg) $_ij d:x' @ self._ns, degree=4)

        self._solu = solver.solve_linear('solu', res, constrain=self._ucons)

        # upscaling
        b = self._domain.integral(self._ns.eval_ij('(phi ks + (1 - phi) kg) ($_ij + du_ij) d:x'), degree=4).eval(
            solu=self._solu)
        psi = self._domain.integral('phi d:x' @ self._ns, degree=2).eval(solu=self._solu)

        # print("Upscaled conductivity = {}".format(b.export("dense")))
        # print("Upscaled porosity = {}".format(psi))

        self._r_nm1 = r

        return b.export("dense"), psi

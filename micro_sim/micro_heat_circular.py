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
        # Constants
        self._lam = 0.1
        self._temperature_eq = 273

        # Elements in one direction
        nelems = 10

        # Set up mesh with periodicity in both X and Y directions
        self._topo, self._geom = mesh.rectilinear([np.linspace(-0.5, 0.5, nelems)] * 2, periodic=(0, 1))
        self._topo_ref = None  # refined topology which is initialized in solve()

        self._ns = function.Namespace()
        self._ns.x = self._geom
        self._ns.basis = self._topo.basis('std', degree=2).vector(2)
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

        self._ucons = np.zeros(len(self._ns.basis), dtype=bool)
        self._ucons[-1] = True  # constrain u to zero at a point

    def _update_radius(self, r, temperature, dt):
        return r + dt * ((temperature ** 2 / self._temperature_eq ** 2) - 1)

    def _phasefield(self, x, y, r):
        phi = 1. / (1. + function.exp(-4. / self._lam * (function.sqrt(x ** 2 + y ** 2) - r)))

        return phi

    def vtk_output(self):
        # Output phase field
        bezier = self._topo.sample('bezier', 2)
        x, phi = bezier.eval(['x_i', 'phi'] @ self._ns)
        with treelog.add(treelog.DataLog()):
            export.vtk('phase-field', bezier.tri, x, phi=phi)

        # u value
        x, u = bezier.eval(['x_i', 'u_i'] @ self._ns, solu=self._solu)
        with treelog.add(treelog.DataLog()):
            export.vtk('u-value', bezier.tri, x, T=u)

    def initialize(self, temperature=273, dt=0.001):
        b, psi = self.solve(temperature, dt)

        return b, psi

    def solve(self, temperature, dt):
        """
        TODO Description
        """
        r = self._update_radius(self._r_nm1, temperature, dt)
        self._ns.phi = self._phasefield(self._ns.x[0], self._ns.x[1], r)

        dist = abs(r - function.norm2(self._geom))
        for margin in self._lam / 2, self._lam / 4, self._lam / 8:
            # refine elements within `margin` of the circle boundary
            active, ielem = self._topo.sample('bezier', 2).eval([margin - dist, self._topo.f_index])
            self._topo_ref = self._topo.refined_by(np.unique(ielem[active > 0]))

        # Define cell problem
        res = self._topo_ref.integral('(phi ks + (1 - phi) kg) u_i,j basis_ni,j d:x' @ self._ns, degree=4)
        res += self._topo_ref.integral('basis_ni,j (phi ks + (1 - phi) kg) $_ij d:x' @ self._ns, degree=4)

        self._solu = solver.solve_linear('solu', res, constrain=self._ucons)

        # upscaling
        b = self._topo_ref.integral(self._ns.eval_ij('(phi ks + (1 - phi) kg) ($_ij + du_ij) d:x'), degree=4).eval(
            solu=self._solu)
        psi = self._topo_ref.integral('phi d:x' @ self._ns, degree=2).eval(solu=self._solu)

        # print("Upscaled conductivity = {}".format(b.export("dense")))
        # print("Upscaled porosity = {}".format(psi))

        self._r_nm1 = r

        return b.export("dense"), psi

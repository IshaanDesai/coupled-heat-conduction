#! /usr/bin/env python3
#
# Micro simulation
# In this script we solve the Laplace equation with a grain depicted by a phase field on a
# square domain :math:`Ω` with boundary :math:`Γ`, subject to periodic
# boundary conditions in both dimensions


from nutils import mesh, function, solver, export, cli
from nutils.sparse import dtype
import treelog
import numpy as np

def temp_rad_linear(T):
    T_min, T_max = 300, 320
    r_min, r_max = 0.1, 0.5

    return r_min + (r_max - r_min) * (T - T_min) / (T_max - T_min)


def phasefield(x, y, r):
    lam = 0.02

    phi = 1. / (1. + function.exp(-4. / lam * (function.sqrt(x ** 2 + y ** 2) - r)))

    return phi


def main(temperature):
    """
    Laplace problem on a unit square.
    """
    # VTK output
    vtk_output = False

    # Log output
    log_output = False

    # Elements in one direction
    nelems = 5

    # Set up mesh with periodicity in both X and Y directions
    domain, geom = mesh.rectilinear([np.linspace(-0.5, 0.5, nelems), np.linspace(-0.5, 0.5, nelems)], periodic=(0, 1))

    ns = function.Namespace()
    ns.x = geom
    ns.basis = domain.basis('std', degree=2).vector(2)
    ns.u = 'basis_ni ?solu_n'
    ns.du_ij = 'u_i,j'

    # Conductivity of grain material
    ns.kg = 5.0
    # Conductivity of sand material
    ns.ks = 1.0

    r = temp_rad_linear(temperature)
    ns.phi = phasefield(ns.x[0], ns.x[1], r)

    if vtk_output:
        # Output phase field
        bezier = domain.sample('bezier', 2)
        x, phi = bezier.eval(['x_i', 'phi'] @ ns)
        with treelog.add(treelog.DataLog()):
            export.vtk('phase-field', bezier.tri, x, phi=phi)

    # Define cell problem
    res = domain.integral('(phi ks + (1 - phi) kg) u_i,j basis_ni,j d:x' @ ns, degree=4)
    res += domain.integral('basis_ni,j (phi ks + (1 - phi) kg) $_ij d:x' @ ns, degree=4)

    ucons = np.zeros(len(ns.basis), dtype=bool)
    ucons[-1] = True  # constrain u to zero at a point

    solu = solver.solve_linear('solu', res, constrain=ucons)

    if vtk_output:
        bezier = domain.sample('bezier', 2)
        x, u = bezier.eval(['x_i', 'u_i'] @ ns, solu=solu)
        with treelog.add(treelog.DataLog()):
            export.vtk('u-value', bezier.tri, x, T=u)

    # upscaling
    b = domain.integral(ns.eval_ij('(phi ks + (1 - phi) kg) ($_ij + du_ij) d:x'), degree=4).eval(solu=solu)
    psi = domain.integral('phi d:x' @ ns, degree=2).eval(solu=solu)

    if log_output:
        print("Upscaled conductivity = {} || Upscaled porosity = {}".format(b.export("dense"), psi))

    return b.export("dense"), psi


if __name__ == '__main__':
    cli.run(main)

#! /usr/bin/env python3
#
# Micro simulation
#
# In this script we solve the Laplace equation :math:`u_{,kk} = 0` and a phase field 
# Allen-Cahn equation on a unit
# square domain :math:`Ω` with boundary :math:`Γ`, subject to periodic
# boundary conditions in both dimensions
#

from nutils import mesh, function, solver, export, cli
from nutils.sparse import dtype
import treelog
import numpy as np


def smoothstep(r):
  edgeh = 0.3 # Radial outer edge of grain interface
  edgel = 0.2 # Radial inner edge of grain interface
  r = (r - edgeh)/(edgeh - edgel)
  
  # function.piecewise(r, [edgel,edgeh], 0, r, 1)

  return r*r*(3 - 2*r)
  
def phasefield(x, y):
  lam = 0.02
  radius = 0.3

  phi = 1. / (1. + function.exp(-4. / lam * function.sqrt(x**2 + y**2) - radius))
  
  return phi 


def main():
  '''
  Laplace problem on a unit square.
  '''
  # Elements in one direction
  nelems = 50

  # Set up mesh with periodicity in both X and Y directions
  domain, geom = mesh.rectilinear([np.linspace(-0.5, 0.5, nelems), np.linspace(-0.5, 0.5, nelems)], periodic=(0,1))

  ns = function.Namespace()
  ns.x = geom
  ns.basis = domain.basis('std', degree=1)
  ns.u = 'basis_n ?lhs_n'
  ns.uwall = 273.0
  ns.uinit = 300.0

  # Conductivity of grain material
  ns.kg = 5.0
  # Conductivity of sand material
  ns.ks = 1.0

  ns.r = 'sqrt(x_i x_i)'

  ns.phi = phasefield(ns.x[0], ns.x[1])
  # ns.phi = smoothstep(ns.r)

  # ns.dphi = function.grad(ns.x, ns.phi)

  # Output phase field
  bezier = domain.sample('bezier', 2)
  x, phi = bezier.eval(['x_i', 'phi'] @ ns)
  with treelog.add(treelog.DataLog()):
    export.vtk('phase-field', bezier.tri, x, phi=phi)

  # Define cell problem 
  res = domain.integral('(phi ks + (1 - phi) kg) u_,i basis_n,i d:x' @ ns, degree=2)
  res -= domain.integral('(phi ks + (1 - phi) kg) basis_n d:x' @ ns, degree=2)

  ucons = np.zeros(len(ns.basis), dtype=bool)
  ucons[-1] = True # constrain u to zero at a point

  lhs = solver.solve_linear('lhs', res, constrain=ucons)

  bezier = domain.sample('bezier', 2)
  x, u = bezier.eval(['x_i', 'u'] @ ns, lhs=lhs)
  with treelog.add(treelog.DataLog()):
    export.vtk('u-value', bezier.tri, x, T=u)

  # upscaling
  b = domain.integral('(phi ks + (1 - phi) kg) u_,i d:x' @ ns, degree=2).eval(lhs=lhs)
  psi = domain.integral('phi d:x' @ ns, degree=2).eval(lhs=lhs)

  return b, psi

if __name__ == '__main__':
  cli.run(main)

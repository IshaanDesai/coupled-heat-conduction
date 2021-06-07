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
import treelog
import numpy as np


def smoothstep(r):
  edgeh = 0.3 # Radial outer edge of grain interface
  edgel = 0.2 # Radial inner edge of grain interface
  r = (r - edgeh)/(edgeh - edgel)
  
  #r = max(r, 0.0)
  #r = min(r, 1.0)

  return r*r*(3 - 2*r)


def main(nelems:int, etype:str, btype:str, degree:int):
  '''
  Laplace problem on a unit square.

  .. arguments::

     nelems [10]
       Number of elements along edge.
     etype [square]
       Type of elements (square/triangle/mixed).
     btype [std]
       Type of basis function (std/spline), availability depending on the
       selected element type.
     degree [1]
       Polynomial degree.
  '''

  # Set up mesh with periodicity in both X and Y directions
  domain, geom = mesh.rectilinear([np.linspace(0, 1, nelems), np.linspace(0, 1, nelems)], periodic=[0, 1])

  ns = function.Namespace()
  ns.x = geom
  ns.basis = domain.basis(btype, degree=degree)
  ns.u = 'basis_n ?lhs_n'
  
  # Conductivity of grain material
  ns.kg = 5.0
  # Conductivity of sand material
  ns.ks = 1.0

  ns.rsharp = 0.375 # Sharp interface of grain
  ns.r = 'sqrt(x_i x_i)'
  ns.phi = smoothstep(ns.r)

  # Define heat equation with phase fields
  # .. math:: ∀ v: ∫_Ω \frac{dv}{dx_i} \frac{d(\phi*k_s + (1-\phi)*k_g)*u}{dx_i} = 0.

  resu = domain.integral('d(basis_n, x_i) d((phi ks + (1 - phi) kg) u, x_i) J(x)' @ ns, degree=degree*2)

  # Boundary conditions
  cons = solver.optimize('lhs', sqr0)

  # No need to define boundary conditions as all boundaries are periodic

  lhs = solver.solve_linear('lhs', resu)

  bezier = domain.sample('bezier', 2)
  x, u = bezier.eval(['x', 'u'] @ ns, lhs=lhs)
  with treelog.add(treelog.DataLog()):
    export.vtk('micro-heat', bezier.tri, x, T=u)

if __name__ == '__main__':
  cli.run(main)

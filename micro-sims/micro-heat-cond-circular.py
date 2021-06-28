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
from math import exp, sqrt


def smoothstep(r):
  edgeh = 0.3 # Radial outer edge of grain interface
  edgel = 0.2 # Radial inner edge of grain interface
  r = (r - edgeh)/(edgeh - edgel)
  
  function.piecewise(r, [0,1], 0, r, 1)

  return r*r*(3 - 2*r)

def phasefieldapprox(x, y):
  # Grain
  lam = 0.02
  radius = 0.3
  grain_x = 0.5
  grain_y = 0.5
  
  return (1. / (1. + exp(-4. / lam * (sqrt( (x-grain_x)**2+(y-grain_y)**2) - (radius+0.001)))))


def main():
  '''
  Laplace problem on a unit square.
  '''
  # Elements in one direction
  nelems = 20

  # Set up mesh with periodicity in both X and Y directions
  # domain, geom = mesh.rectilinear([np.linspace(0, 1, nelems), np.linspace(0, 1, nelems)], periodic=[0, 1])
  domain, geom = mesh.unitsquare(nelems, 'triangle')

  ns = function.Namespace()
  ns.x = geom
  ns.ubasis = domain.basis('std', degree=2).vector(2)
  ns.u_i = 'ubasis_ni ?u_n'
  
  # Conductivity of grain material
  ns.kg = 5.0
  # Conductivity of sand material
  ns.ks = 1.0

  ns.r = 'sqrt(x_i x_i)'
  ns.phi = smoothstep(ns.r)

  # Normal vectors in X and Y directions (outward facing from faces)
  ns.e1 = [1.0, 0.0]
  ns.e2 = [0.0, 1.0]

  usqr = domain.boundary.integral('u_k u_k J(x)' @ ns, degree=4)
  wallcons = solver.optimize('u', usqr, droptol=1e-15)

  # Define cell problem 
  resu = domain.integral('(phi ks + (1 - phi) kg) d(u_i, x_j) d(ubasis_ni, x_j) J(x)' @ ns, degree=4)

  # Forcing term
  resu -= domain.integral('(ks - kg) d(phi, x_j) d(ubasis_ni, x_j) J(x)' @ ns, degree=4)

  lhs = solver.solve_linear(('u'), resu, constrain=wallcons)

  bezier = domain.sample('bezier', 2)
  x, u = bezier.eval(['x_i', 'u_i'] @ ns, lhs=lhs)
  with treelog.add(treelog.DataLog()):
    export.vtk('micro-heat', bezier.tri, x, T=u)

if __name__ == '__main__':
  cli.run(main)

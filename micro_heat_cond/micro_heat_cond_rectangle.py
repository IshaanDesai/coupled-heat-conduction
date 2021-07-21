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


def main():
  '''
  Laplace problem on a unit square.
  '''
  # Elements in one direction
  nelems = 40

  # Set up mesh with periodicity in both X and Y directions
  domain0, geom = mesh.rectilinear([np.linspace(-0.5, 0.5, nelems), np.linspace(-0.5, 0.5, nelems)], periodic=(0,1))

  x, y = geom
  # Cut up a square hole in the middle of the domain
  grainlength = 0.25
  grainheight = 0.25
  domainxcut = domain0.trim(function.abs(x) - grainlength, maxrefine=2)
  domainycut = domain0.trim(function.abs(y) - grainheight, maxrefine=2)

  domain1 = domain0 - domainxcut
  domain2 = domain0 - domainycut

  domaingrain = domain1.subset(domain2)

  domain = domain0 - domain0.subset(domaingrain)

  ns = function.Namespace()
  ns.x = geom
  ns.basis = domain.basis('std', degree=2).vector(2)
  ns.u = 'basis_ni ?solu_n'
  ns.du_ij = 'u_i,j'

  # Define cell problem 
  res = domain.integral('u_i,j basis_ni,j d:x' @ ns, degree=4)
  res += domain.integral('basis_ni,j $_ij d:x' @ ns, degree=4)

  ucons = np.zeros(len(ns.basis), dtype=bool)
  ucons[-1] = True # constrain u to zero at a point

  solu = solver.solve_linear('solu', res, constrain=ucons)

  bezier = domain.sample('bezier', 2)
  x, ui, uj = bezier.eval(['x_i', 'u_i', 'u_j'] @ ns, solu=solu)
  with treelog.add(treelog.DataLog()):
    export.vtk('u-value', bezier.tri, x, Ti=ui, Tj=uj)

  # upscaling
  b = domain.integral(ns.eval_ij('($_ij + du_ij) d:x'), degree=4).eval(solu=solu)
  # psi = domain.integral('phi d:x' @ ns, degree=2).eval(solu=solu)
  psi = 1

  print("Upscaled conductivity = {} || Upscaled porosity = {}".format(b.export("dense"), psi))

  return b.export("dense"), psi

if __name__ == '__main__':
  cli.run(main)

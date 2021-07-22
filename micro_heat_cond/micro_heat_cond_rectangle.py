#! /usr/bin/env python3

from nutils import mesh, function, solver, export, cli
from nutils.sparse import dtype
import treelog
import numpy as np


def main():
  '''
  2D cell problem for diffusion in porous media 
  '''
  # Elements in one direction
  nelems = 40

  # Set up mesh with periodicity in both X and Y directions (unit square mesh)
  domain0, geom = mesh.rectilinear([np.linspace(-0.5, 0.5, nelems), np.linspace(-0.5, 0.5, nelems)], periodic=(0,1))

  # Extract X and Y coordinates of all mesh nodes from the geometry object
  x, y = geom

  # Cut up a rectangular grain in the middle of the domain
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

  # Define a vector basis as the solution is vectorial weights
  ns.basis = domain.basis('std', degree=2).vector(2)

  # Variable "u" are the weights for which the cell problem is solved for
  ns.u = 'basis_ni ?solu_n'
  ns.du_ij = 'u_i,j'

  # Define cell problem in vectorial form
  res = domain.integral('u_i,j basis_ni,j d:x' @ ns, degree=4)
  res += domain.integral('basis_ni,j $_ij d:x' @ ns, degree=4)

  ucons = np.zeros(len(ns.basis), dtype=bool)
  ucons[-1] = True  # constrain u to zero at a point

  # Solve the linear system
  solu = solver.solve_linear('solu', res, constrain=ucons)

  # Generate VTK file output for viewing
  bezier = domain.sample('bezier', 2)
  x, ui, uj = bezier.eval(['x_i', 'u_i', 'u_j'] @ ns, solu=solu)
  with treelog.add(treelog.DataLog()):
    export.vtk('solution', bezier.tri, x, Ti=ui, Tj=uj)

  # Calculating the effective diffusion matrix D
  b = domain.integral(ns.eval_ij('($_ij + du_ij) d:x'), degree=4).eval(solu=solu)

  print("Upscaled conductivity = {}".format(b.export("dense")))


if __name__ == '__main__':
  cli.run(main)

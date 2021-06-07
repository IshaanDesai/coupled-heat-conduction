#! /usr/bin/env python3
#
# In this script we solve the unsteady Heat equation

from nutils import mesh, function, solver, export, cli
import treelog
import numpy as np
import precice

def main(nelems:int, etype:str, btype:str, degree:int, dt:float, endtime:float):
  '''
  Laplace problem on a unit square.

  .. arguments::

     nelems [20]
       Number of elements along edge.
     etype [square]
       Type of elements (square/triangle/mixed).
     btype [std]
       Type of basis function (std/spline), availability depending on the
       selected element type.
     degree [1]
       Polynomial degree.
     dt [.01]
       Time step.
     endtime [1.0]
       End time.
  '''

  domain, geom = mesh.unitsquare(nelems, etype)

  # To be able to write index based tensor contractions, we need to bundle all
  # relevant functions together in a namespace. Here we add the geometry ``x``,
  # a scalar ``basis``, and the solution ``u``. The latter is formed by
  # contracting the basis with a to-be-determined solution vector ``?lhs``.

  ns = function.Namespace()
  ns.x = geom
  ns.basis = domain.basis(btype, degree=degree)
  ns.u = 'basis_n ?lhs_n'
  ns.dt = dt
  ns.dudt = 'basis_n (?lhs_n - ?lhs0_n) / dt' # time derivative
  ns.k = 1.

  # Dirichlet BCs temperatures
  ns.ubottom = 300
  ns.utop = 320

  # Time related variables
  timestep = 0

  # preCICE setup
  interface = precice.Interface("Macro-heat", "./precice-config-xml", 0, 1)

  # define coupling meshes
  meshName = "macro-mesh"
  meshID = interface.get_mesh_id(meshName)

  # Define Gauss points on entire domain as coupling mesh
  couplingsample = domain.sample('gauss', degree=2)  # mesh located at Gauss points
  dataIndices = interface.set_mesh_vertices(meshID, couplingsample.eval(ns.x0, ns.x1))

  # coupling data
  readData = "Conductivity"
  readdataID = interface.get_data_id(readData, meshID)

  # initialize preCICE
  precice_dt = interface.initialize()
  dt = min(precice_dt, dt)

  # define the weak form
  res = domain.integral('(basis_n dudt + basis_n,i u_,i) d:x' @ ns, degree=2)

  # Set Dirichlet boundary conditions
  sqr = domain.boundary['bottom'].integral('(u - ubottom)^2 d:x' @ ns, degree=2)
  sqr += domain.boundary['top'].integral('(u - utop)^2 d:x' @ ns, degree=2)
  cons = solver.optimize('lhs', sqr, droptol=1e-15)

  # No need to add Neumann boundary condition for right and left boundaries
  # as they are adiabatic walls, hence flux = 0

  cons0 = cons  # to not lose the Dirichlet BC at the bottom
  lhs0 = np.zeros(res.shape)  # solution from previous timestep

  # set Dirichlet BCs as initial conditions and visualize initial state
  sqr0 = domain.boundary['bottom'].integral('(u - ubottom)^2' @ ns, degree=2)
  sqr0 += domain.boundary['top'].integral('(u - utop)^2' @ ns, degree=2)
  lhs0 = solver.optimize('lhs', sqr0)

  bezier = domain.sample('bezier', 2)
  x, u = bezier.eval(['x_i', 'u'] @ ns, lhs=lhs0)
  with treelog.add(treelog.DataLog()):
    export.vtk('Solid_0', bezier.tri, x, T=u)

  # time loop
  while interface.is_coupling_ongoing():

    # read conductivity values from interface
    if interface.is_read_data_available():
      readdata = interface.read_block_scalar_data(readdataID, dataIndices)
      coupledata = couplingsample.asfunction(readdata)
      sqr = couplingsample.integral(((ns.k - coupledata)**2).sum(0))

      # solve timestep
      lhs = solver.solve_linear('lhs', res, constrain=cons, arguments=dict(lhs0=lhs0, dt=dt))

      # do the coupling
      precice_dt = interface.advance(dt)
      dt = min(precice_dt, dt)

      # advance variables
      timestep += 1
      lhs0 = lhs

      # visualization
      if timestep % 20 == 0:  # visualize
        bezier = domain.sample('bezier', 2)
        x, u = bezier.eval(['x_i', 'u'] @ ns, lhs=lhs)
        with treelog.add(treelog.DataLog()):
          export.vtk('macro-heat-' + str(timestep), bezier.tri, x, T=u)

  interface.finalize()

if __name__ == '__main__':
  cli.run(main)

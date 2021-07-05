#! /usr/bin/env python3
#
# In this script we solve the unsteady Heat equation

from nutils import mesh, function, solver, export, cli
import treelog
import numpy as np
import precice
from config import Config

def main():
  '''
  2D unsteady heat equation on a unit square.
  The material consists of a mixture of two materials, the grain and sand 
  '''
  # Elements in one direction
  nelems = 20

  domain, geom = mesh.unitsquare(nelems, 'square')

  config = Config("macro-config.json") 

  coupling = config.is_coupling_on()

  ns = function.Namespace()
  ns.x = geom
  ns.basis = domain.basis('std', degree=1)
  ns.u = 'basis_n ?lhs_n'

  if coupling:
    # Coupling quantities
    ns.phi = 'basis_n ?solphi_n'
    ns.k = 'basis_n ?solk_n'
  else:
    ns.phi = 0.5
    ns.k = 1.0
  
  phi = 0.5 # initial value
  k = 1.0 # initial value

  ns.rhos = 1.0
  ns.rhog = 2.0
  ns.dudt = '(rhos phi + (1 - phi) rhog) basis_n (?lhs_n - ?lhs0_n) / ?dt'

  # Dirichlet BCs temperatures
  ns.ubottom = 300
  ns.utop = 320

  # Time related variables
  ns.dt = config.get_dt()
  n = 0
  t = 0
  dt = config.get_dt()
  end_t = config.get_total_time()

  if coupling:
    # preCICE setup
    interface = precice.Interface(config.get_participant_name(), config.get_config_file_name(), 0, 1)

    # define coupling meshes
    readMeshName = config.get_read_mesh_name()
    readMeshID = interface.get_mesh_id(readMeshName)

    # Define Gauss points on entire domain as coupling mesh
    couplingsample = domain.sample('gauss', degree=2)  # mesh located at Gauss points
    vertex_ids = interface.set_mesh_vertices(readMeshID, couplingsample.eval(ns.x))
    print("n_vertices on macro domain = {}".format(vertex_ids.size))

    sqrphi = couplingsample.integral((ns.phi - phi)**2)
    solphi = solver.optimize('solphi', sqrphi, droptol=1E-12)

    sqrk = couplingsample.integral((ns.k - k)**2)
    solk = solver.optimize('solk', sqrk, droptol=1E-12)

    # coupling data
    readDataName = config.get_read_data_name()
    read_cond_id = interface.get_data_id(readDataName[0], readMeshID)
    read_poro_id = interface.get_data_id(readDataName[1], readMeshID)

    # initialize preCICE
    precice_dt = interface.initialize()
    dt = function.min(precice_dt, ns.dt)

  # define the weak form
  res = domain.integral('(basis_n dudt + k basis_n,i u_,i) d:x' @ ns, degree=2)

  # Set Dirichlet boundary conditions
  sqr = domain.boundary['bottom'].integral('(u - ubottom)^2 d:x' @ ns, degree=2)
  sqr += domain.boundary['top'].integral('(u - utop)^2 d:x' @ ns, degree=2)
  cons = solver.optimize('lhs', sqr, droptol=1e-15)

  # No need to add Neumann boundary conditions for right and left boundaries
  # as they are adiabatic walls, hence flux = 0

  lhs0 = np.zeros(res.shape)  # solution from previous timestep
  # set u = ubottom and visualize 
  sqr = domain.integral('(u - ubottom)^2' @ ns, degree=2)
  lhs0 = solver.optimize('lhs', sqr)
  bezier = domain.sample('bezier', 2)
  x, u = bezier.eval(['x_i', 'u'] @ ns, lhs=lhs0)
  with treelog.add(treelog.DataLog()):
    export.vtk('macro-heat-' + str(n), bezier.tri, x, T=u)

  # time loop
  while t < end_t:
    if coupling:
      # read conductivity values from interface
      if interface.is_read_data_available():
        # Read porosity and apply
        print("vertex_ids = {}".format(vertex_ids))
        poro_data = interface.read_block_scalar_data(read_poro_id, vertex_ids)
        print("poro_data from preCICE = {}".format(poro_data))
        poro_coupledata = couplingsample.asfunction(poro_data)
        sqrphi = couplingsample.integral((ns.phi - poro_coupledata)**2)
        solphi = solver.optimize('solphi', sqrphi, droptol=1E-12)

        # Read conductivity and apply
        cond_data = interface.read_block_scalar_data(read_cond_id, vertex_ids)
        cond_coupledata = couplingsample.asfunction(cond_data)
        sqrk = couplingsample.integral((ns.k - cond_coupledata)**2)
        solk = solver.optimize('solk', sqrk, droptol=1E-12)

    if coupling:
      # solve timestep
      lhs = solver.solve_linear('lhs', res, constrain=cons, arguments=dict(lhs0=lhs0, dt=dt, solphi=solphi, solk=solk))
      print("before solving")
    else:
      lhs = solver.solve_linear('lhs', res, constrain=cons, arguments=dict(lhs0=lhs0, dt=dt))

    if coupling:
      # do the coupling
      precice_dt = interface.advance(dt)
      dt = function.min(precice_dt, dt)

    # advance variables
    n += 1
    t = n*dt
    lhs0 = lhs

    # visualization
    if n % 20 == 0:  # visualize
      bezier = domain.sample('bezier', 2)
      x, u = bezier.eval(['x_i', 'u'] @ ns, lhs=lhs)
      with treelog.add(treelog.DataLog()):
        export.vtk('macro-heat-' + str(n), bezier.tri, x, T=u)

  if coupling:
    interface.finalize()

if __name__ == '__main__':
  cli.run(main)

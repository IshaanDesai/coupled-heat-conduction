#! /usr/bin/env python3
#
# In this script we solve the unsteady Heat equation

from nutils import mesh, function, solver, export, cli
import treelog
import numpy as np
import precice
from config import Config


def main():
    """
    2D unsteady heat equation on a unit square.
    The material consists of a mixture of two materials, the grain and sand
    """
    # Elements in one direction
    nelems = 5

    topo, geom = mesh.unitsquare(nelems, 'square')

    config = Config("macro-config.json")

    ns = function.Namespace(fallback_length=2)
    ns.x = geom
    ns.basis = topo.basis('std', degree=1)
    ns.kbasis = topo.basis('std', degree=1).vector(topo.ndims).vector(topo.ndims)
    ns.u = 'basis_n ?solu_n'

    # Coupling quantities
    ns.phi = 'basis_n ?solphi_n'
    ns.k_ij = 'kbasis_nij ?solk_n'

    phi = 0.5  # initial value
    k = 1.0  # initial value

    ns.rhos = 1.0
    ns.rhog = 2.0
    ns.dudt = 'basis_n (?solu_n - ?solu0_n) / ?dt'

    # Dirichlet BCs temperatures
    ns.ubottom = 273
    ns.utop = 370

    # Time related variables
    ns.dt = config.get_dt()
    n = 0
    t = 0
    dt = config.get_dt()
    t_end = config.get_total_time()
    n_end = int(t_end / dt)
    t_out = config.get_t_output()
    n_out = int(t_out / dt)

    # preCICE setup
    interface = precice.Interface(config.get_participant_name(), config.get_config_file_name(), 0, 1)

    # define coupling meshes
    read_mesh_name = config.get_read_mesh_name()
    read_mesh_id = interface.get_mesh_id(read_mesh_name)
    write_mesh_name = config.get_write_mesh_name()
    write_mesh_id = interface.get_mesh_id(write_mesh_name)

    # Define Gauss points on entire domain as coupling mesh
    couplingsample = topo.sample('gauss', degree=2)  # mesh located at Gauss points
    vertex_ids = interface.set_mesh_vertices(read_mesh_id, couplingsample.eval(ns.x))

    sqrphi = couplingsample.integral((ns.phi - phi) ** 2)
    solphi = solver.optimize('solphi', sqrphi, droptol=1E-12)

    sqrk = couplingsample.integral(((ns.k - k * np.eye(2)) * (ns.k - k * np.eye(2))).sum([0, 1]))
    solk = solver.optimize('solk', sqrk, droptol=1E-12)

    # coupling data
    read_data_name = config.get_read_data_name()
    k_00_id = interface.get_data_id(read_data_name[0], read_mesh_id)
    k_01_id = interface.get_data_id(read_data_name[1], read_mesh_id)
    k_10_id = interface.get_data_id(read_data_name[2], read_mesh_id)
    k_11_id = interface.get_data_id(read_data_name[3], read_mesh_id)

    poro_id = interface.get_data_id(read_data_name[4], read_mesh_id)

    write_data_name = config.get_write_data_name()
    temperature_id = interface.get_data_id(write_data_name, write_mesh_id)

    # initialize preCICE
    precice_dt = interface.initialize()
    dt = min(precice_dt, dt)

    # define the weak form
    res = topo.integral('((rhos phi + (1 - phi) rhog) basis_n dudt + k_ij basis_n,i u_,j) d:x' @ ns, degree=2)

    # Set Dirichlet boundary conditions
    sqr = topo.boundary['bottom'].integral('(u - ubottom)^2 d:x' @ ns, degree=2)
    sqr += topo.boundary['top'].integral('(u - utop)^2 d:x' @ ns, degree=2)
    cons = solver.optimize('solu', sqr, droptol=1e-15)

    # Set domain to initial condition
    sqr = topo.integral('(u - ubottom)^2' @ ns, degree=2)
    solu0 = solver.optimize('solu', sqr)

    if interface.is_action_required(precice.action_write_initial_data()):
        temperatures = couplingsample.eval('u' @ ns, solu=solu0)
        interface.write_block_scalar_data(temperature_id, vertex_ids, temperatures)

        interface.mark_action_fulfilled(precice.action_write_initial_data())

    interface.initialize_data()

    # Prepare the post processing sample
    bezier = topo.sample('bezier', 2)

    # VTK output of initial state
    x, phi, u = bezier.eval(['x_i', 'phi', 'u'] @ ns, solphi=solphi, solu=solu0)
    with treelog.add(treelog.DataLog()):
        export.vtk('macro-heat-initial', bezier.tri, x, T=u)

    # time loop
    while interface.is_coupling_ongoing():
        # write checkpoint
        if interface.is_action_required(precice.action_write_iteration_checkpoint()):
            solu_checkpoint = solu0
            t_checkpoint = t
            n_checkpoint = n
            interface.mark_action_fulfilled(precice.action_write_iteration_checkpoint())

        # Read porosity and apply
        poro_data = interface.read_block_scalar_data(poro_id, vertex_ids)
        poro_coupledata = couplingsample.asfunction(poro_data)

        sqrphi = couplingsample.integral((ns.phi - poro_coupledata) ** 2)
        solphi = solver.optimize('solphi', sqrphi, droptol=1E-12)

        # Read conductivity and apply
        k_00 = interface.read_block_scalar_data(k_00_id, vertex_ids)
        k_01 = interface.read_block_scalar_data(k_01_id, vertex_ids)
        k_10 = interface.read_block_scalar_data(k_10_id, vertex_ids)
        k_11 = interface.read_block_scalar_data(k_11_id, vertex_ids)

        k_00_c = couplingsample.asfunction(k_00)
        k_01_c = couplingsample.asfunction(k_01)
        k_10_c = couplingsample.asfunction(k_10)
        k_11_c = couplingsample.asfunction(k_11)

        k_coupledata = function.asarray([[k_00_c, k_01_c], [k_10_c, k_11_c]])
        sqrk = couplingsample.integral(((ns.k - k_coupledata) * (ns.k - k_coupledata)).sum([0, 1]))
        solk = solver.optimize('solk', sqrk, droptol=1E-12)

        # solve timestep
        solu = solver.solve_linear('solu', res, constrain=cons,
                                   arguments=dict(solu0=solu0, dt=dt, solphi=solphi, solk=solk))

        temperatures = couplingsample.eval('u' @ ns, solu=solu)
        interface.write_block_scalar_data(temperature_id, vertex_ids, temperatures)

        # do the coupling
        precice_dt = interface.advance(dt)
        dt = min(precice_dt, dt)

        # advance variables
        n += 1
        t += dt
        solu0 = solu

        if interface.is_action_required(precice.action_read_iteration_checkpoint()):
            solu0 = solu_checkpoint
            t = t_checkpoint
            n = n_checkpoint
            interface.mark_action_fulfilled(precice.action_read_iteration_checkpoint())
        else:  # go to next timestep
            if n % n_out == 0 or n == n_end:  # visualize
                x, phi, u = bezier.eval(['x_i', 'phi', 'u'] @ ns, solphi=solphi, solu=solu)
                with treelog.add(treelog.DataLog()):
                    export.vtk('macro-heat-' + str(n), bezier.tri, x, T=u, phi=phi)

    interface.finalize()


if __name__ == '__main__':
    cli.run(main)

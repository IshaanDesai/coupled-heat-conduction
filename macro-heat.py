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

    domain, geom = mesh.unitsquare(nelems, 'square')

    config = Config("macro-config.json")

    coupling = config.is_coupling_on()

    ns = function.Namespace(fallback_length=2)
    ns.x = geom
    ns.basis = domain.basis('std', degree=1)
    ns.kbasis = domain.basis('std', degree=1).vector(2).vector(2)
    ns.u = 'basis_n ?lhs_n'

    if coupling:
        # Coupling quantities
        ns.phi = 'basis_n ?solphi_n'
        ns.k_ij = 'kbasis_nij ?solk_n'
    else:
        ns.phi = 0.5
        ns.k = 1.0

    phi = 0.5  # initial value
    k = 1.0  # initial value

    ns.rhos = 1.0
    ns.rhog = 2.0
    ns.dudt = '(rhos phi + (1 - phi) rhog) basis_n (?lhs_n - ?lhs0_n) / ?dt'

    # Dirichlet BCs temperatures
    ns.ubottom = 273
    ns.utop = 330

    # Time related variables
    ns.dt = config.get_dt()
    n = 0
    t = 0
    dt = config.get_dt()
    t_end = config.get_total_time()
    n_end = int(t_end / dt)
    t_out = config.get_t_output()
    n_out = int(t_out / dt)

    if coupling:
        # preCICE setup
        interface = precice.Interface(config.get_participant_name(), config.get_config_file_name(), 0, 1)

        # define coupling meshes
        read_mesh_name = config.get_read_mesh_name()
        read_mesh_id = interface.get_mesh_id(read_mesh_name)
        write_mesh_name = config.get_write_mesh_name()
        write_mesh_id = interface.get_mesh_id(write_mesh_name)

        # Define Gauss points on entire domain as coupling mesh
        couplingsample = domain.sample('gauss', degree=2)  # mesh located at Gauss points
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
    res = domain.integral('(basis_n dudt + k_ij basis_n,i u_,j) d:x' @ ns, degree=2)

    # Set Dirichlet boundary conditions
    sqr = domain.boundary['bottom'].integral('(u - ubottom)^2 d:x' @ ns, degree=2)
    sqr += domain.boundary['top'].integral('(u - utop)^2 d:x' @ ns, degree=2)
    cons = solver.optimize('lhs', sqr, droptol=1e-15)

    # Set domain to initial condition
    sqr = domain.integral('(u - ubottom)^2' @ ns, degree=2)
    lhs0 = solver.optimize('lhs', sqr)

    if coupling:
        if interface.is_action_required(precice.action_write_initial_data()):
            temperatures = couplingsample.eval('u' @ ns, lhs=lhs0)
            interface.write_block_scalar_data(temperature_id, vertex_ids, temperatures)

            interface.mark_action_fulfilled(precice.action_write_initial_data())
        interface.initialize_data()

    # Prepare the post processing sample
    bezier = domain.sample('bezier', 2)

    # VTK output of initial state
    x, u = bezier.eval(['x_i', 'u'] @ ns, lhs=lhs0)
    with treelog.add(treelog.DataLog()):
        export.vtk('macro-heat-' + str(n), bezier.tri, x, T=u)

    # time loop
    while interface.is_coupling_ongoing():
        if coupling:
            # write checkpoint
            if interface.is_action_required(precice.action_write_iteration_checkpoint()):
                lhs_checkpoint = lhs0
                t_checkpoint = t
                n_checkpoint = n
                interface.mark_action_fulfilled(precice.action_write_iteration_checkpoint())

            # read conductivity values from interface
            if interface.is_read_data_available():
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
        if coupling:
            lhs = solver.solve_linear('lhs', res, constrain=cons,
                                      arguments=dict(lhs0=lhs0, dt=dt, solphi=solphi, solk=solk))

            temperatures = couplingsample.eval('u' @ ns, lhs=lhs)
            interface.write_block_scalar_data(temperature_id, vertex_ids, temperatures)
        else:
            lhs = solver.solve_linear('lhs', res, constrain=cons, arguments=dict(lhs0=lhs0, dt=dt))

        if coupling:
            # do the coupling
            precice_dt = interface.advance(dt)
            dt = min(precice_dt, dt)

        # advance variables
        n += 1
        t += dt
        lhs0 = lhs

        if interface.is_action_required(precice.action_read_iteration_checkpoint()):
            lhs0 = lhs_checkpoint
            t = t_checkpoint
            n = n_checkpoint
            interface.mark_action_fulfilled(precice.action_read_iteration_checkpoint())
        else: # go to next timestep
            # visualization
            if n % n_out == 0 or n == n_end:  # visualize
                x, u = bezier.eval(['x_i', 'u'] @ ns, lhs=lhs)
                with treelog.add(treelog.DataLog()):
                    export.vtk('macro-heat-' + str(n), bezier.tri, x, T=u)

    if coupling:
        interface.finalize()


if __name__ == '__main__':
    cli.run(main)

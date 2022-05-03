#! /usr/bin/env python3
#

import numpy as np
import precice
from config import Config


def main():
    """
    Dummy macro simulation which is coupled to a set of micro simulations via preCICE and the Micro Manager
    """
    config = Config("macro-dummy-config.json")

    nv = 25

    n = n_checkpoint = 0
    t = t_checkpoint = 0
    dt = config.get_dt()
    t_end = config.get_total_time()
    n_end = int(t_end / dt)

    # preCICE setup
    interface = precice.Interface(config.get_participant_name(), config.get_config_file_name(), 0, 1)

    # define coupling meshes
    read_mesh_name = config.get_read_mesh_name()
    read_mesh_id = interface.get_mesh_id(read_mesh_name)
    write_mesh_name = config.get_write_mesh_name()
    write_mesh_id = interface.get_mesh_id(write_mesh_name)

    # Coupling mesh
    coords = np.empty([nv, 2])
    coords_x = coords_y = np.arange(0, 1, 0.2)
    count = 0
    for x in coords_x:
        for y in coords_y:
            coords[count, 0] = x
            coords[count, 1] = y
            count += 1

    # Define Gauss points on entire domain as coupling mesh
    vertex_ids = interface.set_mesh_vertices(read_mesh_id, coords)

    # coupling data
    read_data_name = config.get_read_data_name()
    read_data_id = interface.get_data_id(read_data_name, read_mesh_id)

    write_data_name = config.get_write_data_name()
    write_data_id = interface.get_data_id(write_data_name, write_mesh_id)

    # initialize preCICE
    precice_dt = interface.initialize()
    dt = min(precice_dt, dt)

    write_scalar_data = np.zeros([nv])
    write_vector_data = np.zeros([nv, 2])

    for i in range(nv):
        write_scalar_data[i] = i
        write_vector_data[i, 0] = i
        write_vector_data[i, 1] = i + nv

    if interface.is_action_required(precice.action_write_initial_data()):
        interface.write_block_scalar_data(write_data_id, vertex_ids, write_scalar_data)
        interface.write_block_vector_data(write_data_id, vertex_ids, write_vector_data)
        interface.mark_action_fulfilled(precice.action_write_initial_data())

    interface.initialize_data()

    # time loop
    while interface.is_coupling_ongoing():
        # write checkpoint
        if interface.is_action_required(precice.action_write_iteration_checkpoint()):
            checkpoint = "checkpoint"
            t_checkpoint = t
            n_checkpoint = n
            interface.mark_action_fulfilled(precice.action_write_iteration_checkpoint())

        # Read porosity and apply
        read_data = interface.read_block_scalar_data(read_data_id, vertex_ids)

        write_scalar_data[:] = read_data[:]
        write_vector_data[:, 0] = read_data[:]
        write_vector_data[:, 1] = read_data[:] + 1

        interface.write_block_scalar_data(write_data_id, vertex_ids, write_scalar_data)
        interface.write_block_vector_data(write_data_id, vertex_ids, write_vector_data)

        # do the coupling
        precice_dt = interface.advance(dt)
        dt = min(precice_dt, dt)

        # advance variables
        n += 1
        t += dt

        if interface.is_action_required(precice.action_read_iteration_checkpoint()):
            # old_state = checkpoint
            t = t_checkpoint
            n = n_checkpoint
            interface.mark_action_fulfilled(precice.action_read_iteration_checkpoint())

    interface.finalize()


if __name__ == '__main__':
    main()

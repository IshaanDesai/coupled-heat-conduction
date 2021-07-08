#! /usr/bin/env python3
#
# Micro manager to couple a macro code to multiple micro codes

import numpy as np
import precice
from config import Config
from micro_heat_cond.micro_heat_cond_circular import main
from nutils import mesh

# Elements in one direction
nelems = 20
domain, geom = mesh.unitsquare(nelems, 'square')

config = Config("micro-manager-config.json")

dt = config.get_dt()

interface = precice.Interface(config.get_participant_name(), config.get_config_file_name(), 0, 1)

# define coupling mesh
writeMeshName = config.get_write_mesh_name()
writeMeshID = interface.get_mesh_id(writeMeshName)

# Define Gauss points on entire domain as coupling mesh
couplingsample = domain.sample('gauss', degree=2)  # mesh located at Gauss points
vertex_ids = interface.set_mesh_vertices(writeMeshID, couplingsample.eval(geom))
print("n_vertices in micro manager = {}".format(vertex_ids.size))

# coupling data
writeDataName = config.get_write_data_name()
write_cond_id = interface.get_data_id(writeDataName[0], writeMeshID)
write_poro_id = interface.get_data_id(writeDataName[1], writeMeshID)

# initialize preCICE
precice_dt = interface.initialize()
dt = min(precice_dt, dt)

if interface.is_action_required(precice.action_write_initial_data()):
    # Solve micro simulations
    k, phi = main()

    # Assemble data to write to preCICE
    # k_vals = np.full(vertex_ids.size, (k[0] + k[1]) / 2.)
    k_vals = np.full(vertex_ids.size, k)
    phi_vals = np.full(vertex_ids.size, phi)

    # write data
    interface.write_block_scalar_data(write_cond_id, vertex_ids, k_vals)
    interface.write_block_scalar_data(write_poro_id, vertex_ids, phi_vals)

    interface.mark_action_fulfilled(precice.action_write_initial_data())

interface.initialize_data()

while interface.is_coupling_ongoing():
    # Solve micro simulations
    k, phi = main()

    # Assemble data to write to preCICE
    # k_vals = np.full(vertex_ids.size, (k[0] + k[1]) / 2.)
    phi_vals = np.full(vertex_ids.size, phi)
    k_vals = np.full(vertex_ids.size, k)

    # write data
    interface.write_block_scalar_data(write_poro_id, vertex_ids, phi_vals)
    interface.write_block_scalar_data(write_cond_id, vertex_ids, k_vals)

    # do the coupling
    precice_dt = interface.advance(dt)
    dt = min(precice_dt, dt)

interface.finalize()

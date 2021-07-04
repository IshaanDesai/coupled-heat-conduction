#! /usr/bin/env python3
#
# Micro manager to couple a macro code to multiple micro codes

import numpy as np
import precice
from config import Config
from micro_heat_cond.micro_heat_cond_circular import main

# Number of Gauss points in one direction
nelems = 20
dx = 1. / 20

config = Config("micro-manager-config.json")

dt = config.get_dt()

interface = precice.Interface(config.get_participant_name(), config.get_config_file_name(), 0, 1)

# define coupling mesh
writeMeshName = config.get_write_mesh_name()
writeMeshID = interface.get_mesh_id(writeMeshName)

# Generate a coordinate grid identical to macro mesh
coords = []
for i in range(nelems + 1):
    for j in range(nelems + 1):
        coords.append([i*dx, j*dx]) # uniform grid

vertex_ids = interface.set_mesh_vertices(writeMeshID, coords)

# coupling data
writeDataName = config.get_write_data_name()
write_cond_id = interface.get_data_id(writeDataName[0], writeMeshID)
write_poro_id = interface.get_data_id(writeDataName[1], writeMeshID)

# initialize preCICE
precice_dt = interface.initialize()
dt = min(precice_dt, dt)

while interface.is_coupling_ongoing():
    # Solve micro simulations
    k, phi = main()

    # Assemble data to write to preCICE
    k_vals = np.full((nelems*nelems), k[0])
    phi_vals = np.full((nelems*nelems), phi)

    # write data
    interface.write_block_scalar_data(write_cond_id, vertex_ids, k_vals)
    interface.write_block_scalar_data(write_poro_id, vertex_ids, phi_vals)

    # do the coupling
    precice_dt = interface.advance(dt)
    dt = min(precice_dt, dt)

interface.finalize()

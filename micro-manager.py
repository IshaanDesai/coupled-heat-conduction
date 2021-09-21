#! /usr/bin/env python3
#
# Micro manager to couple a macro code to multiple micro codes

import numpy as np
import precice
from config import Config
from micro_sim.micro_heat_circular import main
from nutils import mesh
from mpi4py import MPI

# MPI related variables
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def slice_tensor(a):
    a_00, a_01, a_10, a_11 = [], [], [], []
    for i in range(len(a)):
        a_00.append(a[i][0][0])
        a_01.append(a[i][0][1])
        a_10.append(a[i][1][0])
        a_11.append(a[i][1][1])

    return a_00, a_01, a_10, a_11


# Elements in one direction
nelems = 10

domain, geom = mesh.unitsquare(nelems, 'square')

config = Config("micro-manager-config.json")

dt = config.get_dt()

interface = precice.Interface(config.get_participant_name(), config.get_config_file_name(), rank, size)

# coupling mesh names
writeMeshName = config.get_write_mesh_name()
writeMeshID = interface.get_mesh_id(writeMeshName)
readMeshName = config.get_read_mesh_name()
readMeshID = interface.get_mesh_id(readMeshName)

# Define bounding box with extents of entire macro mesh
dx = (1 - 0) / size

# All processes except last process get equal area of domain
macroMeshBounds = [rank * dx, (rank + 1) * dx, 0, 1] if rank < size - 1 else [rank * dx, 1, 0, 1]

interface.set_mesh_access_region(writeMeshID, macroMeshBounds)

# coupling data
writeDataName = config.get_write_data_name()
k_00_id = interface.get_data_id(writeDataName[0], writeMeshID)
k_01_id = interface.get_data_id(writeDataName[1], writeMeshID)
k_10_id = interface.get_data_id(writeDataName[2], writeMeshID)
k_11_id = interface.get_data_id(writeDataName[3], writeMeshID)

poro_id = interface.get_data_id(writeDataName[4], writeMeshID)

readDataName = config.get_read_data_name()
temperature_id = interface.get_data_id(readDataName, readMeshID)

# initialize preCICE
precice_dt = interface.initialize()
dt = min(precice_dt, dt)

macroVertexIDs, macroVertexCoords = interface.get_mesh_vertices_and_ids(writeMeshID)

while interface.is_coupling_ongoing():
    # Read temperature values from preCICE
    if interface.is_read_data_available():
        temperatures = interface.read_block_scalar_data(temperature_id, macroVertexIDs)

    k = []
    phi = []
    print("Rank {} is solving micro simulations...".format(rank))
    for T in temperatures:
        k_i, phi_i = main(T)
        k.append(k_i)
        phi.append(phi_i)

    # Reformat conductivity tensor into arrays of component-wise scalars
    k_00, k_01, k_10, k_11 = slice_tensor(k)

    # Write conductivity and porosity to preCICE
    interface.write_block_scalar_data(k_00_id, macroVertexIDs, k_00)
    interface.write_block_scalar_data(k_01_id, macroVertexIDs, k_01)
    interface.write_block_scalar_data(k_01_id, macroVertexIDs, k_01)
    interface.write_block_scalar_data(k_10_id, macroVertexIDs, k_10)
    interface.write_block_scalar_data(k_11_id, macroVertexIDs, k_11)
    interface.write_block_scalar_data(poro_id, macroVertexIDs, phi)

    precice_dt = interface.advance(dt)
    dt = min(precice_dt, dt)

interface.finalize()

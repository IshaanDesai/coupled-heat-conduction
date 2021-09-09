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

# define coupling mesh
writeMeshName = config.get_write_mesh_name()
writeMeshID = interface.get_mesh_id(writeMeshName)
readMeshName = config.get_read_mesh_name()
readMeshID = interface.get_mesh_id(readMeshName)

# Define Gauss points on entire domain as coupling mesh
couplingsample = domain.sample('gauss', degree=2)  # mesh located at Gauss points

coords_global = couplingsample.eval(geom)
nv_global, _ = coords_global.shape
nv_local = int(nv_global / size)

coupling_coords = []
# All processes except the last process get equal number of vertices
if rank < size - 1:
    for i in range(rank * nv_local, (rank + 1) * nv_local):
        coupling_coords.append(coords_global[i])

# Last process gets its share and remaining vertices if any
if rank == size - 1:
    for i in range(rank * nv_local, nv_global):
        coupling_coords.append(coords_global[i])

coupling_coords = np.array(coupling_coords)

vertex_ids = interface.set_mesh_vertices(writeMeshID, coupling_coords)

# coupling data
writeDataName = config.get_write_data_name()
k_00_id = interface.get_data_id(writeDataName[0], writeMeshID)
k_01_id = interface.get_data_id(writeDataName[1], writeMeshID)
k_10_id = interface.get_data_id(writeDataName[2], writeMeshID)
k_11_id = interface.get_data_id(writeDataName[3], writeMeshID)

poro_id = interface.get_data_id(writeDataName[4], writeMeshID)

readDataName = config.get_read_data_name()
grain_rad_id = interface.get_data_id(readDataName, readMeshID)

# initialize preCICE
precice_dt = interface.initialize()
dt = min(precice_dt, dt)

if interface.is_action_required(precice.action_write_initial_data()):
    # Solve micro simulations
    k, phi = main()

    # Assemble data to write to preCICE
    k_00 = np.full(vertex_ids.size, k[0][0])
    k_01 = np.full(vertex_ids.size, k[0][1])
    k_10 = np.full(vertex_ids.size, k[1][0])
    k_11 = np.full(vertex_ids.size, k[1][1])
    phi_vals = np.full(vertex_ids.size, phi)

    # write data
    interface.write_block_scalar_data(k_00_id, vertex_ids, k_00)
    interface.write_block_scalar_data(k_01_id, vertex_ids, k_01)
    interface.write_block_scalar_data(k_10_id, vertex_ids, k_10)
    interface.write_block_scalar_data(k_11_id, vertex_ids, k_11)

    interface.write_block_scalar_data(poro_id, vertex_ids, phi_vals)

    interface.mark_action_fulfilled(precice.action_write_initial_data())

if interface.is_read_data_available:
    # Read grain radius from preCICE
    grain_rads = interface.read_block_scalar_data(grain_rad_id, vertex_ids)

k = []
phi = []
# Solve micro problems
for r in grain_rads:
    k_i, phi_i = main(r)
    k.append(k_i)
    phi.append(phi_i)

while interface.is_coupling_ongoing():
    k_00, k_01, k_10, k_11 = slice_tensor(k)

    # write data
    interface.write_block_scalar_data(k_00_id, vertex_ids, k_00)
    interface.write_block_scalar_data(k_01_id, vertex_ids, k_01)
    interface.write_block_scalar_data(k_10_id, vertex_ids, k_10)
    interface.write_block_scalar_data(k_11_id, vertex_ids, k_11)

    interface.write_block_scalar_data(poro_id, vertex_ids, phi)

    # do the coupling
    precice_dt = interface.advance(dt)
    dt = min(precice_dt, dt)

interface.finalize()

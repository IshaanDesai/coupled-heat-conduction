#! /usr/bin/env python3
#
# Micro manager to couple a macro code to multiple micro codes

import numpy as np
import precice
from config import Config
from micro_heat_cond.micro_heat_cond_circular import main
from nutils import mesh
from mpi4py import MPI

# MPI related variables
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

grain_rad_all = None
nv = 0

if rank == 0:
    # Elements in one direction
    nelems = 10
    domain, geom = mesh.unitsquare(nelems, 'square')

    config = Config("micro-manager-config.json")

    dt = config.get_dt()

    interface = precice.Interface(config.get_participant_name(), config.get_config_file_name(), 0, 1)

    # define coupling mesh
    writeMeshName = config.get_write_mesh_name()
    writeMeshID = interface.get_mesh_id(writeMeshName)
    readMeshName = config.get_read_mesh_name()
    readMeshID = interface.get_mesh_id(readMeshName)

    # Define Gauss points on entire domain as coupling mesh
    couplingsample = domain.sample('gauss', degree=2)  # mesh located at Gauss points
    vertex_ids = interface.set_mesh_vertices(writeMeshID, couplingsample.eval(geom))
    nv = vertex_ids.size

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
        grain_rad_all = interface.read_block_scalar_data(grain_rad_id, vertex_ids)

nv = comm.bcast(nv, root=0)
nv_l = int(nv/size)

grain_rad = np.zeros(nv_l)

# Scatter grain radius values to all processes
comm.Scatter(grain_rad_all, grain_rad, root=0)

k_local = []
phi_local = np.zeros(nv_l)
# Solve micro problems
for i in range(nv_l):
    k_i, phi_i = main(grain_rad[i])
    k_local.append(k_i)
    phi_local[i] = phi_i

k_local = np.array(k_local)

print("Rank: {} k_local = {}".format(rank, k_local))

if rank == 0:
    k = np.zeros(nv)
    phi = np.zeros(nv)
else:
    k = None
    phi = None

# Gather conductivity and porosity values from all processes
comm.Gather(k_local, k, root=0)
comm.Gather(phi_local, phi, root=0)

if rank == 0:
    while interface.is_coupling_ongoing():
        # Break up the tensor into 1D scalar data array for writing to preCICE
        k_00 = k[:][0][0]
        k_01 = k[:][0][1]
        k_10 = k[:][1][0]
        k_11 = k[:][1][1]

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

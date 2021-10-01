"""
Micro manager to couple a macro code to multiple micro codes
"""

import precice
from config import Config
from micro_sim.micro_heat_circular import MicroSimulation
from mpi4py import MPI

# MPI related variables
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def write_block_tensor_data(tensor, solver_interface, data_ids, vertex_ids):
    a_00, a_01, a_10, a_11 = [], [], [], []
    for x in range(len(tensor)):
        a_00.append(tensor[x][0][0])
        a_01.append(tensor[x][0][1])
        a_10.append(tensor[x][1][0])
        a_11.append(tensor[x][1][1])

    solver_interface.write_block_scalar_data(data_ids[0], vertex_ids, a_00)
    solver_interface.write_block_scalar_data(data_ids[1], vertex_ids, a_01)
    solver_interface.write_block_scalar_data(data_ids[2], vertex_ids, a_10)
    solver_interface.write_block_scalar_data(data_ids[3], vertex_ids, a_11)


config = Config("micro-manager-config.json")

dt = config.get_dt()
t_out = config.get_t_output()
n_out = int(t_out / dt)

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
writeDataNames = config.get_write_data_name()
writeDataIDs = []
for name in writeDataNames:
    writeDataIDs.append(interface.get_data_id(name, writeMeshID))

readDataNames = config.get_read_data_name()
readDataIDs = []
for name in readDataNames:
    readDataIDs.append(interface.get_data_id(name, readMeshID))

# initialize preCICE
precice_dt = interface.initialize()
dt = min(precice_dt, dt)

macroVertexIDs, macroVertexCoords = interface.get_mesh_vertices_and_ids(writeMeshID)
nv, _ = macroVertexCoords.shape

micro_sims = []
for v in range(nv):
    micro_sims.append(MicroSimulation())

k, phi = [], []
for v in range(nv):
    k_i, phi_i = micro_sims[v].initialize(dt=dt)
    k.append(k_i)
    phi.append(phi_i)

micro_sims[0].vtk_output()

writeData = []
# Initialize coupling data
if interface.is_action_required(precice.action_write_initial_data()):
    write_block_tensor_data(k, interface, writeDataIDs, macroVertexIDs)
    interface.write_block_scalar_data(writeDataIDs[4], macroVertexIDs, phi)

    interface.mark_action_fulfilled(precice.action_write_initial_data())

interface.initialize_data()

t, n = 0, 0

while interface.is_coupling_ongoing():
    # Write checkpoint
    if interface.is_action_required(precice.action_write_iteration_checkpoint()):
        k_checkpoint = k
        phi_checkpoint = phi
        n_checkpoint = n
        interface.mark_action_fulfilled(precice.action_write_iteration_checkpoint())

    # Read temperature values from preCICE
    for data_id in readDataIDs:
        readData = interface.read_block_scalar_data(data_id, macroVertexIDs)

    print("Rank {} is solving micro simulations...".format(rank))
    k, phi = [], []
    i = 0
    for data in readData:
        k_i, phi_i = micro_sims[i].solve(temperature=data, dt=dt)
        k.append(k_i)
        phi.append(phi_i)
        i += 1

    write_block_tensor_data(k, interface, writeDataIDs, macroVertexIDs)
    interface.write_block_scalar_data(writeDataIDs[4], macroVertexIDs, phi)

    precice_dt = interface.advance(dt)
    dt = min(precice_dt, dt)

    t += dt
    n += 1

    if interface.is_action_required(precice.action_read_iteration_checkpoint()):
        k = k_checkpoint
        phi = phi_checkpoint
        n = n_checkpoint
        interface.mark_action_fulfilled(precice.action_read_iteration_checkpoint())
    else:
        if n == n_out:
            micro_sims[0].vtk_output()

interface.finalize()

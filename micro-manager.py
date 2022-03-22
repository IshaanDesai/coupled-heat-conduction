"""
Micro manager to couple a macro code to multiple micro codes
"""

import precice
from config import Config
from micro_sim.micro_heat_circular import MicroSimulation
from mpi4py import MPI
from math import sqrt

# MPI related variables
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def write_block_matrix_data(m, solver_interface, data_ids, vertex_ids):
    a_00, a_01, a_10, a_11 = [], [], [], []
    for x in range(len(m)):
        a_00.append(m[x][0][0])
        a_01.append(m[x][0][1])
        a_10.append(m[x][1][0])
        a_11.append(m[x][1][1])

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

macro_bounds = config.get_macro_domain_bounds()

# Bounds of macro domain
macro_xmin = macro_bounds[0]
macro_xmax = macro_bounds[1]
macro_ymin = macro_bounds[2]
macro_ymax = macro_bounds[3]

# Domain decomposition
size_x = int(sqrt(size))
while size % size_x != 0:
    size_x -= 1

size_y = int(size / size_x)

if rank == 0:
    print("Partitions in X direction: {}".format(size_x))
    print("Partitions in Y direction: {}".format(size_y))

dx = (macro_xmax - macro_xmin) / size_x
dy = (macro_ymax - macro_ymin) / size_y

local_xmin = dx * (rank % size_x)
local_ymin = dy * int(rank / size_x)

macroMeshBounds = [local_xmin, local_xmin + dx, local_ymin, local_ymin + dy]
print("Rank {}: Macro mesh bounds {}".format(rank, macroMeshBounds))

interface.set_mesh_access_region(writeMeshID, macroMeshBounds)

# Configure data written to preCICE
writeData = dict()
writeDataNames = config.get_write_data_name()
writeDataIDs = []
for name in writeDataNames:
    writeDataIDs.append(interface.get_data_id(name, writeMeshID))
    writeData[name] = []

# Configure data read from preCICE
readData = dict()
readDataNames = config.get_read_data_name()
readDataIDs = []
for name in readDataNames:
    readDataIDs.append(interface.get_data_id(name, readMeshID))
    readData[name] = []

# initialize preCICE
precice_dt = interface.initialize()
dt = min(precice_dt, dt)

# Get macro mesh from preCICE
macroVertexIDs, macroVertexCoords = interface.get_mesh_vertices_and_ids(writeMeshID)
nv, _ = macroVertexCoords.shape

# Create all micro simulation objects
micro_sims = []
for v in range(nv):
    micro_sims.append(MicroSimulation())

k, phi = [], []
i = 0
for v in range(nv):
    micro_sims_output = micro_sims[v].initialize(dt=dt)
    for data in micro_sims_output:
        writeData[writeDataNames[i]].append(data)
        i += 1

writeData = []
# Initialize coupling data
if interface.is_action_required(precice.action_write_initial_data()):
    write_block_matrix_data(k, interface, writeDataIDs, macroVertexIDs)
    interface.write_block_scalar_data(writeDataIDs[4], macroVertexIDs, phi)

    interface.mark_action_fulfilled(precice.action_write_initial_data())

interface.initialize_data()

t, n = 0, 0
t_checkpoint, n_checkpoint = 0, 0

while interface.is_coupling_ongoing():
    # Write checkpoint
    if interface.is_action_required(precice.action_write_iteration_checkpoint()):
        for v in range(nv):
            micro_sims[v].save_checkpoint()

        t_checkpoint = t
        n_checkpoint = n
        interface.mark_action_fulfilled(precice.action_write_iteration_checkpoint())

    # Read temperature values from preCICE
    for data_id in readDataIDs:
        readData = interface.read_block_scalar_data(data_id, macroVertexIDs)

    print("Rank {} is solving micro simulations...".format(rank))
    micro_sims_output = []
    i = 0
    for data in readData:
        micro_sims_output.append(micro_sims[i].solve(temperature=data, dt=dt))

    write_block_matrix_data(k, interface, writeDataIDs, macroVertexIDs)
    interface.write_block_scalar_data(writeDataIDs[4], macroVertexIDs, phi)

    precice_dt = interface.advance(dt)
    dt = min(precice_dt, dt)

    t += dt
    n += 1

    if interface.is_action_required(precice.action_read_iteration_checkpoint()):
        for v in range(nv):
            micro_sims[v].revert_to_checkpoint()

        n = n_checkpoint
        t = t_checkpoint
        interface.mark_action_fulfilled(precice.action_read_iteration_checkpoint())

interface.finalize()

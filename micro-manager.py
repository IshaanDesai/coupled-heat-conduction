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

# Bounds of macro domain
macro_xmin = 0
macro_xmax = 1
macro_ymin = 0
macro_ymax = 1

# Domain decomposition
assert size % 2 == 0, "Only even number of processors are permissible"

# Define bounding box with extents of entire macro mesh
dx = (1 - 0) / size

# All processes except last process get equal area of domain
macroMeshBounds = [rank * dx, (rank + 1) * dx, 0, 1] if rank < size - 1 else [rank * dx, 1, 0, 1]

interface.set_mesh_access_region(writeMeshID, macroMeshBounds)

# Configure data written to preCICE
writeDataNames = config.get_write_data_name()
writeDataIDs = []
for name in writeDataNames:
    writeDataIDs.append(interface.get_data_id(name, writeMeshID))

# Configure data read from preCICE
readDataNames = config.get_read_data_name()
readDataIDs = []
for name in readDataNames:
    readDataIDs.append(interface.get_data_id(name, readMeshID))

# initialize preCICE
precice_dt = interface.initialize()
dt = min(precice_dt, dt)

# Get macro mesh from preCICE
macroVertexIDs, macroVertexCoords = interface.get_mesh_vertices_and_ids(writeMeshID)
nv, _ = macroVertexCoords.shape

# Initialize all micro simulations
micro_sims = []
for v in range(nv):
    micro_sims.append(MicroSimulation())

k, phi = [], []
for v in range(nv):
    k_i, phi_i = micro_sims[v].initialize(dt=dt)
    k.append(k_i)
    phi.append(phi_i)

# micro_sims[0].vtk_output(rank)

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
            micro_sims[v].save_state()

        t_checkpoint = t
        n_checkpoint = n
        interface.mark_action_fulfilled(precice.action_write_iteration_checkpoint())

    # Read temperature values from preCICE
    for data_id in readDataIDs:
        readData = interface.read_block_scalar_data(data_id, macroVertexIDs)

    print("Rank {} is solving micro simulations...".format(rank))
    k, phi = [], []
    i = 0
    for data in readData:
        phi_i = micro_sims[i].solve_allen_cahn(temperature=data, dt=dt)
        phi.append(phi_i)

        k_i = micro_sims[i].solve_heat_cell_problem()
        k.append(k_i)

        micro_sims[i].refine_mesh()

        i += 1

    write_block_matrix_data(k, interface, writeDataIDs, macroVertexIDs)
    interface.write_block_scalar_data(writeDataIDs[4], macroVertexIDs, phi)

    precice_dt = interface.advance(dt)
    dt = min(precice_dt, dt)

    t += dt
    n += 1

    if interface.is_action_required(precice.action_read_iteration_checkpoint()):
        for v in range(nv):
            micro_sims[v].revert_state()

        n = n_checkpoint
        t = t_checkpoint
        interface.mark_action_fulfilled(precice.action_read_iteration_checkpoint())
    # else:
    #     if n % n_out == 0:
    #         micro_sims[0].vtk_output(rank)

interface.finalize()

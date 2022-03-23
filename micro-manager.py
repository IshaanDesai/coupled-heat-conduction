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


def write_data_to_precice(solver_interface, data_ids, vertex_ids, data):
    for name in data_ids.keys():
        if isinstance(data[name][0], list):
            assert size(data[name][0]) == 2, "Vector data to be written to preCICE has incorrect dimensions"
            solver_interface.write_block_vector_data(data_ids[name], vertex_ids, data[name])
        else:
            solver_interface.write_block_scalar_data(data_ids[name], vertex_ids, data[name])


config = Config("micro-manager-config.json")

dt = config.get_dt()
t_out = config.get_t_output()
n_out = int(t_out / dt)

interface = precice.Interface(config.get_participant_name(), config.get_config_file_name(), rank, size)

# coupling mesh names
write_mesh_name = config.get_write_mesh_name()
write_mesh_id = interface.get_mesh_id(write_mesh_name)
read_mesh_name = config.get_read_mesh_name()
read_mesh_id = interface.get_mesh_id(read_mesh_name)

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

mesh_bounds = [local_xmin, local_xmin + dx, local_ymin, local_ymin + dy]
print("Rank {}: Macro mesh bounds {}".format(rank, mesh_bounds))

interface.set_mesh_access_region(write_mesh_id, mesh_bounds)

# Configure data written to preCICE
write_data = dict()
write_data_names = config.get_write_data_name()
write_data_ids = dict()
for name in write_data_names:
    write_data_ids[name] = interface.get_data_id(name, write_mesh_id)
    write_data[name] = []

# Configure data read from preCICE
read_data = dict()
read_data_names = config.get_read_data_name()
readDataIDs = []
for name in read_data_names:
    readDataIDs.append(interface.get_data_id(name, read_mesh_id))
    read_data[name] = []

# initialize preCICE
precice_dt = interface.initialize()
dt = min(precice_dt, dt)

# Get macro mesh from preCICE
mesh_vertex_ids, mesh_vertex_coords = interface.get_mesh_vertices_and_ids(write_mesh_id)
nv, _ = mesh_vertex_coords.shape

print("Rank {}: Macro vertex coords {}".format(rank, mesh_vertex_coords))

# Create all micro simulation objects
micro_sims = []
for v in range(nv):
    micro_sims.append(MicroSimulation())

k, phi = [], []
i = 0
for v in range(nv):
    micro_sims_output = micro_sims[v].initialize()
    for data in micro_sims_output:
        write_data[write_data_names[i]].append(data)
        i += 1

# Initialize coupling data
if interface.is_action_required(precice.action_write_initial_data()):
    write_data_to_precice(interface, write_data_ids, mesh_vertex_ids, write_data)
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
        readData = interface.read_block_scalar_data(data_id, mesh_vertex_ids)

    micro_sims_output = []
    i = 0
    for data in readData:
        micro_sims_output.append(micro_sims[i].solve(data, dt))

    write_data_to_precice(interface, write_data_ids, mesh_vertex_ids, micro_sims_output)

    precice_dt = interface.advance(dt)
    dt = min(precice_dt, dt)

    t += dt
    n += 1

    if interface.is_action_required(precice.action_read_iteration_checkpoint()):
        for v in range(nv):
            micro_sims[v].reload_checkpoint()

        n = n_checkpoint
        t = t_checkpoint
        interface.mark_action_fulfilled(precice.action_read_iteration_checkpoint())

interface.finalize()

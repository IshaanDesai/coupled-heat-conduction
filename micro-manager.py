"""
Micro manager to couple a macro code to multiple micro codes
"""

import sys
import precice
from config import Config
from mpi4py import MPI
from math import sqrt

# MPI related variables
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def write_data_to_precice(solver_interface, data_ids, vertex_ids, data):
    for dname in data_ids.keys():
        if isinstance(data[dname][0], list):
            assert size(data[dname][0]) == 2, "Vector data to be written to preCICE has incorrect dimensions"
            solver_interface.write_block_vector_data(data_ids[dname], vertex_ids, data[dname])
        else:
            solver_interface.write_block_scalar_data(data_ids[dname], vertex_ids, data[dname])


config_file_name = ''.join(sys.argv[1:])
config = Config(config_file_name)

micro_file_name = config.get_micro_file_name()
MicroSimulation = getattr(__import__(micro_file_name, fromlist=["MicroSimulation"]), "MicroSimulation")

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

dx = (macro_xmax - macro_xmin) / size_x
dy = (macro_ymax - macro_ymin) / size_y

local_xmin = dx * (rank % size_x)
local_ymin = dy * int(rank / size_x)

mesh_bounds = [local_xmin, local_xmin + dx, local_ymin, local_ymin + dy]
print("Rank {}: Macro mesh bounds {}".format(rank, mesh_bounds))

interface.set_mesh_access_region(write_mesh_id, mesh_bounds)

# Configure data written to preCICE
write_data = dict()
write_data_ids = dict()
write_data_names = config.get_write_data_name()
if isinstance(write_data_names, list):
    for name in write_data_names:
        write_data_ids[name] = interface.get_data_id(name, write_mesh_id)
        write_data[name] = []
else:
    write_data_ids[write_data_names] = interface.get_data_id(write_data_names, write_mesh_id)
    write_data[write_data_names] = []

# Configure data read from preCICE
read_data = dict()
read_data_ids = dict()
read_data_names = config.get_read_data_name()
if isinstance(read_data_names, list):
    for name in read_data_names:
        read_data_ids[name] = (interface.get_data_id(name, read_mesh_id))
        read_data[name] = []
else:
    read_data_ids[read_data_names] = interface.get_data_id(read_data_names, read_mesh_id)
    read_data[read_data_names] = []

# initialize preCICE
precice_dt = interface.initialize()
dt = min(precice_dt, dt)

# Get macro mesh from preCICE
mesh_vertex_ids, mesh_vertex_coords = interface.get_mesh_vertices_and_ids(write_mesh_id)
nms, _ = mesh_vertex_coords.shape

print("Rank {}: Macro vertex coords {}".format(rank, mesh_vertex_coords))

# Create all micro simulation objects
micro_sims = []
for v in range(nms):
    micro_sims.append(MicroSimulation())

k, phi = [], []
if hasattr(MicroSimulation, 'initialize') and callable(getattr(MicroSimulation, 'initialize')):
    for v in range(nms):
        i = 0
        micro_sims_output = micro_sims[v].initialize()
        if micro_sims_output is not None:
            for data in micro_sims_output:
                if isinstance(write_data_names, list):
                    write_data[write_data_names[i]].append(data)
                    i += 1
                else:
                    write_data[write_data_names].append(data)
        else:
            if isinstance(write_data_names, list):
                for name in write_data_names:
                    write_data[name].append(0.0)
            else:
                write_data[write_data_names].append(0.0)


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
        for v in range(nms):
            micro_sims[v].save_checkpoint()

        t_checkpoint = t
        n_checkpoint = n
        interface.mark_action_fulfilled(precice.action_write_iteration_checkpoint())

    for name in read_data_ids.keys():
        read_data = interface.read_block_scalar_data(read_data_ids[name], mesh_vertex_ids)

    micro_sims_output = []
    i = 0
    for data in read_data:
        micro_sims_output.append(micro_sims[i].solve(data, dt))
        i += 1

    i = 0
    for key in write_data.keys():
        write_data[key] = []

    for data in micro_sims_output:
        if isinstance(write_data_names, list):
            write_data[write_data_names[i]].append(data)
            i += 1
        else:
            write_data[write_data_names].append(data)

    write_data_to_precice(interface, write_data_ids, mesh_vertex_ids, write_data)

    precice_dt = interface.advance(dt)
    dt = min(precice_dt, dt)

    t += dt
    n += 1

    if interface.is_action_required(precice.action_read_iteration_checkpoint()):
        for v in range(nms):
            micro_sims[v].reload_checkpoint()

        n = n_checkpoint
        t = t_checkpoint
        interface.mark_action_fulfilled(precice.action_read_iteration_checkpoint())

interface.finalize()

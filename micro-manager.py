"""
Micro manager to couple a macro code to multiple micro codes
"""

import sys
import precice
from micro_manager_tools.config import Config
from mpi4py import MPI
from math import sqrt
import numpy as np
from functools import reduce
from operator import iconcat

# MPI related variables
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

assert len(sys.argv) == 2, "Incorrect run command. The micro-manager is run as: python micro-manager.py <config-file>"

config_file_name = ''.join(sys.argv[1:])
print("Provided configuration file: {}".format(config_file_name))
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

assert len(macro_bounds) / 2 == interface.get_dimensions(), "Provided macro mesh bounds are of incorrect dimension"

# Domain decomposition
size_x = int(sqrt(size))
while size % size_x != 0:
    size_x -= 1

size_y = int(size / size_x)

dx = abs(macro_bounds[0] - macro_bounds[1]) / size_x
dy = abs(macro_bounds[2] - macro_bounds[3]) / size_y

local_xmin = dx * (rank % size_x)
local_ymin = dy * int(rank / size_x)

mesh_bounds = []
if interface.get_dimensions() == 2:
    mesh_bounds = [local_xmin, local_xmin + dx, local_ymin, local_ymin + dy]
elif interface.get_dimensions() == 3:
    # TODO: Domain needs to be decomposed optimally in the Z direction too
    mesh_bounds = [local_xmin, local_xmin + dx, local_ymin, local_ymin + dy, macro_bounds[4], macro_bounds[5]]

interface.set_mesh_access_region(write_mesh_id, mesh_bounds)

# Configure data written to preCICE
write_data = dict()
write_data_ids = dict()
write_data_names = config.get_write_data_name()
assert isinstance(write_data_names, dict)
for name in write_data_names.keys():
    write_data_ids[name] = interface.get_data_id(name, write_mesh_id)
    write_data[name] = []

# Configure data read from preCICE
read_data = dict()
read_data_ids = dict()
read_data_names = config.get_read_data_name()
assert isinstance(read_data_names, dict)
for name in read_data_names.keys():
    read_data_ids[name] = interface.get_data_id(name, read_mesh_id)
    read_data[name] = []

# initialize preCICE
precice_dt = interface.initialize()
dt = min(precice_dt, dt)

# Get macro mesh from preCICE (API function is experimental)
mesh_vertex_ids, mesh_vertex_coords = interface.get_mesh_vertices_and_ids(write_mesh_id)
nms, _ = mesh_vertex_coords.shape

# Create all micro simulations
micro_sims = []
for v in range(nms):
    micro_sims.append(MicroSimulation())

# Initialize all micro simulations
if hasattr(MicroSimulation, 'initialize') and callable(getattr(MicroSimulation, 'initialize')):
    for v in range(nms):
        micro_sims_output = micro_sims[v].initialize()
        if micro_sims_output is not None:
            for data_name, data in micro_sims_output.items():
                write_data[data_name].append(data)
        else:
            for name, dim in write_data_names.items():
                if dim == 0:
                    write_data[name].append(0.0)
                elif dim == 1:
                    write_data[name].append(np.zeros(interface.get_dimensions()))

# Initialize coupling data
if interface.is_action_required(precice.action_write_initial_data()):
    for dname, dim in write_data_names.items():
        if dim == 1:
            interface.write_block_vector_data(write_data_ids[dname], mesh_vertex_ids, write_data[dname])
        elif dim == 0:
            interface.write_block_scalar_data(write_data_ids[dname], mesh_vertex_ids, write_data[dname])
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

    for name, dims in read_data_names.items():
        if dims == 0:
            read_data.update({name: interface.read_block_scalar_data(read_data_ids[name], mesh_vertex_ids)})
        elif dims == 1:
            read_data.update({name: interface.read_block_vector_data(read_data_ids[name], mesh_vertex_ids)})

    micro_sims_input = [dict(zip(read_data, t)) for t in zip(*read_data.values())]
    micro_sims_output = []
    for i in range(nms):
        micro_sims_output.append(micro_sims[i].solve(micro_sims_input[i], dt))

    # write_data = {k: reduce(iconcat, [dic[k] for dic in micro_sims_output], []) for k in micro_sims_output[0]}

    write_data = dict()
    for name in micro_sims_output[0]:
        write_data[name] = []

    for dic in micro_sims_output:
        for name, values in dic.items():
            write_data[name].append(values)

    for dname, dim in write_data_names.items():
        if dim == 0:
            print("Scalar write_data: {}".format(write_data[dname]))
            interface.write_block_scalar_data(write_data_ids[dname], mesh_vertex_ids, write_data[dname])
        elif dim == 1:
            print("Vector write_data: {}".format(write_data[dname]))
            interface.write_block_vector_data(write_data_ids[dname], mesh_vertex_ids, write_data[dname])

    precice_dt = interface.advance(dt)
    dt = min(precice_dt, dt)

    t += dt
    n += 1

    # Read checkpoint if required
    if interface.is_action_required(precice.action_read_iteration_checkpoint()):
        for v in range(nms):
            micro_sims[v].reload_checkpoint()

        n = n_checkpoint
        t = t_checkpoint
        interface.mark_action_fulfilled(precice.action_read_iteration_checkpoint())

interface.finalize()

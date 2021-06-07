#! /usr/bin/env python3
#
# Micro manager to couple a macro code to multiple micro codes

import numpy as np
import precice

# Number of Gauss points in one direction
nelems = 100

dt = 1.0e-4

interface = precice.Interface("Micro-manager", "./precice-config.xml", 0, 1)

# define coupling mesh
meshName = "micro-sims"
meshID = interface.get_mesh_id(meshName)

microIDs = np.arange(nelems*nelems)

# Define coordinates of Gauss points on which micro-sims live
# This is already available in the macro-sim, any idea to do this smartly?
# Micro-sims do not care where they are in the macro perspective

# coupling data
readData = "Conductivity"
readdataID = interface.get_data_id(readData, meshID)

# initialize preCICE
precice_dt = interface.initialize()
dt = min(precice_dt, dt)

# Create micro simulations

while interface.is_coupling_ongoing():
    # Solve micro simulations

    # do the coupling
    precice_dt = interface.advance(dt)
    dt = min(precice_dt, dt)

    # write data to interface

interface.finalize()

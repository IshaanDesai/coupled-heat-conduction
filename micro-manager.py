#! /usr/bin/env python3
#
# Micro manager to couple a macro code to multiple micro codes

import numpy as np
import precice

nelems = 10

# At every point of macro simulation there is a micro simulation
microIDs = 0

# Call Micro simulations one after the other
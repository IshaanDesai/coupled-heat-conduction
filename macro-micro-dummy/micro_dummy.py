"""
Micro simulation
In this script we solve a dummy micro problem to just show the working of the macro-micro coupling
"""
import math

from nutils import mesh, function, solver, export, sample, cli
import treelog
import numpy as np


class MicroSimulation:

    def __init__(self):
        """
        Constructor of MicroSimulation class.
        """
        self._dims = 2
        self._micro_scalar_data = None
        self._micro_vector_data = None
        self._checkpoint = None

    def initialize(self):
        self._micro_scalar_data = 0
        self._micro_vector_data = np.zeros([self._dims])
        self._checkpoint = 0

    def solve(self, macro_data, dt):
        assert dt != 0
        self._micro_scalar_data = macro_data["macro-data"] + 1
        self._micro_vector_data[0]

        return {"micro-data": [self._micro_data.copy()]}

    def save_checkpoint(self):
        self._checkpoint = self._micro_data

    def reload_checkpoint(self):
        self._micro_data = self._checkpoint

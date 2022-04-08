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

        self._micro_data = None
        self._checkpoint = None

    def initialize(self):
        self._micro_data = 0
        self._checkpoint = 0
        print("Micro simulation is initialized")

    def solve(self, macro_data, dt):
        print("Solving micro problem with dt = {}".format(dt))
        self._micro_data = macro_data + 1

        return self._micro_data.copy()

    def save_checkpoint(self):
        self._checkpoint = self._micro_data

    def reload_checkpoint(self):
        self._micro_data = self._checkpoint

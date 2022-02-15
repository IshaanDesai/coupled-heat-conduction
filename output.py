"""
This module provides functions to output the data in VTK and other formats for viewing and further post processing
For the VTK export the following package is used: https://github.com/paulo-herrera/PyEVTK
"""

from evtk.hl import pointsToVTK
import numpy as np
import csv


class Output:
    def __init__(self, filename, rank, coords):
        self._rank = rank
        self._coords_x = np.ascontiguousarray(coords[:, 0])
        self._coords_y = np.ascontiguousarray(coords[:, 1])
        self._coords_z = np.ascontiguousarray(np.zeros_like(coords[:, 0]))
        nv = self._coords_x.shape
        print("VTK data of {} vertices will be written".format(nv))
        self._filename = filename
        self._name = []  # List of names of output variables

    def set_output_variable(self, name):
        self._name.append(name)

    def write_vtk(self, n, *fields):
        assert len(self._name) == len(fields), "Number of output variables provided is incorrect"

        dataset = []
        for field in fields:
            dataset.append(np.ascontiguousarray(field, dtype=np.float32))

        filename = self._filename + "_{}_{}".format(self._rank, n)
        pointsToVTK("./output/" + filename, self._coords_x, self._coords_y, self._coords_z,
                    data={self._name[i]: dataset[i] for i in range(len(self._name))})

    def write_csv(self, n, field):
        n_vertices = self._coords_x.shape

        with open('./output/' + self._filename + '_' + str(n) + '.csv', mode='w') as file:
            file_writer = csv.writer(file, delimiter=',')
            for i in range(n_vertices):
                file_writer.writerow([self._coords[i, 0], self._coords[i, 1], field[i]])

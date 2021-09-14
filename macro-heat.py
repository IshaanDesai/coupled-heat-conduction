#! /usr/bin/env python3
#
# In this script we solve the unsteady Heat equation

from nutils import mesh, function, solver, export, cli
import treelog
import numpy as np
import precice
from config import Config


def main():
    """
    2D unsteady heat equation on a unit square.
    The material consists of a mixture of two materials, the grain and sand
    """
    # Elements in one direction
    nelems = 5

    domain, geom = mesh.unitsquare(nelems, 'square')

    config = Config("macro-config.json")

    coupling = config.is_coupling_on()

    ns = function.Namespace(fallback_length=2)
    ns.x = geom
    ns.basis = domain.basis('std', degree=1)
    ns.kbasis = domain.basis('std', degree=1).vector(2).vector(2)
    ns.u = 'basis_n ?lhs_n'

    if coupling:
        # Coupling quantities
        ns.phi = 'basis_n ?solphi_n'
        ns.k_ij = 'kbasis_nij ?solk_n'
    else:
        ns.phi = 0.5
        ns.k = 1.0

    phi = 0.5  # initial value
    k = 1.0  # initial value

    ns.rhos = 1.0
    ns.rhog = 2.0
    ns.dudt = '(rhos phi + (1 - phi) rhog) basis_n (?lhs_n - ?lhs0_n) / ?dt'

    # Dirichlet BCs temperatures
    ns.ubottom = 300
    ns.utop = 320

    # Time related variables
    ns.dt = config.get_dt()
    n = 0
    t = 0
    dt = config.get_dt()
    end_t = config.get_total_time()

    if coupling:
        # preCICE setup
        interface = precice.Interface(config.get_participant_name(), config.get_config_file_name(), 0, 1)

        # define coupling meshes
        readMeshName = config.get_read_mesh_name()
        readMeshID = interface.get_mesh_id(readMeshName)
        writeMeshName = config.get_write_mesh_name()
        writeMeshID = interface.get_mesh_id(writeMeshName)

        # Define Gauss points on entire domain as coupling mesh
        couplingsample = domain.sample('gauss', degree=2)  # mesh located at Gauss points
        vertex_ids = interface.set_mesh_vertices(readMeshID, couplingsample.eval(ns.x))

        sqrphi = couplingsample.integral((ns.phi - phi) ** 2)
        solphi = solver.optimize('solphi', sqrphi, droptol=1E-12)

        sqrk = couplingsample.integral(((ns.k - k * np.eye(2)) * (ns.k - k * np.eye(2))).sum([0, 1]))
        solk = solver.optimize('solk', sqrk, droptol=1E-12)

        # coupling data
        readDataName = config.get_read_data_name()
        k_00_id = interface.get_data_id(readDataName[0], readMeshID)
        k_01_id = interface.get_data_id(readDataName[1], readMeshID)
        k_10_id = interface.get_data_id(readDataName[2], readMeshID)
        k_11_id = interface.get_data_id(readDataName[3], readMeshID)

        poro_id = interface.get_data_id(readDataName[4], readMeshID)

        writeDataName = config.get_write_data_name()
        grain_rad_id = interface.get_data_id(writeDataName, writeMeshID)

        # initialize preCICE
        precice_dt = interface.initialize()
        dt = min(precice_dt, dt)

        interface.initialize_data()

        grain_rads = []
        grain_rad_0 = 0.3
        for v in couplingsample.eval(ns.x):
            grain_rads.append(grain_rad_0 * abs(abs(v[1]) - 0.5) / 0.5)

    # define the weak form
    res = domain.integral('(basis_n dudt + k_ij basis_n,i u_,j) d:x' @ ns, degree=2)

    # Set Dirichlet boundary conditions
    sqr = domain.boundary['bottom'].integral('(u - ubottom)^2 d:x' @ ns, degree=2)
    sqr += domain.boundary['top'].integral('(u - utop)^2 d:x' @ ns, degree=2)
    cons = solver.optimize('lhs', sqr, droptol=1e-15)

    lhs0 = np.zeros(res.shape)  # solution from previous timestep

    # set u = ubottom and visualize
    sqr = domain.integral('(u - ubottom)^2' @ ns, degree=2)
    lhs0 = solver.optimize('lhs', sqr)
    bezier = domain.sample('bezier', 2)
    x, u = bezier.eval(['x_i', 'u'] @ ns, lhs=lhs0)
    with treelog.add(treelog.DataLog()):
        export.vtk('macro-heat-' + str(n), bezier.tri, x, T=u)

    # time loop
    while t < end_t:
        if coupling:
            # read conductivity values from interface
            if interface.is_read_data_available():
                # Read porosity and apply
                poro_data = interface.read_block_scalar_data(poro_id, vertex_ids)
                poro_coupledata = couplingsample.asfunction(poro_data)
                sqrphi = couplingsample.integral((ns.phi - poro_coupledata) ** 2)
                solphi = solver.optimize('solphi', sqrphi, droptol=1E-12)

                # Read conductivity and apply
                k_00 = interface.read_block_scalar_data(k_00_id, vertex_ids)
                k_01 = interface.read_block_scalar_data(k_01_id, vertex_ids)
                k_10 = interface.read_block_scalar_data(k_10_id, vertex_ids)
                k_11 = interface.read_block_scalar_data(k_11_id, vertex_ids)

                k_00_c = couplingsample.asfunction(k_00)
                k_01_c = couplingsample.asfunction(k_01)
                k_10_c = couplingsample.asfunction(k_10)
                k_11_c = couplingsample.asfunction(k_11)

                k_coupledata = function.asarray([[k_00_c, k_01_c], [k_10_c, k_11_c]])
                sqrk = couplingsample.integral(((ns.k - k_coupledata) * (ns.k - k_coupledata)).sum([0, 1]))
                solk = solver.optimize('solk', sqrk, droptol=1E-12)

        # solve timestep
        if coupling:
            lhs = solver.solve_linear('lhs', res, constrain=cons,
                                      arguments=dict(lhs0=lhs0, dt=dt, solphi=solphi, solk=solk))

            interface.write_block_scalar_data(grain_rad_id, vertex_ids, grain_rads)
        else:
            lhs = solver.solve_linear('lhs', res, constrain=cons, arguments=dict(lhs0=lhs0, dt=dt))

        if coupling:
            # do the coupling
            precice_dt = interface.advance(dt)
            dt = min(precice_dt, dt)

        # advance variables
        n += 1
        t = n * dt
        lhs0 = lhs

        # visualization
        if n % 20 == 0:  # visualize
            bezier = domain.sample('bezier', 2)
            x, u = bezier.eval(['x_i', 'u'] @ ns, lhs=lhs)
            with treelog.add(treelog.DataLog()):
                export.vtk('macro-heat-' + str(n), bezier.tri, x, T=u)

    if coupling:
        interface.finalize()


if __name__ == '__main__':
    cli.run(main)
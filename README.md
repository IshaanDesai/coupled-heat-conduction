# coupled-heat-conduction

<a style="text-decoration: none" href="https://github.com/precice/fenics-adapter/blob/master/LICENSE" target="_blank">
    <img src="https://img.shields.io/github/license/IshaanDesai/coupled-heat-conduction.svg" alt="GNU LGPL license">
</a>

This code solves a heat conduction problem on a 2D domain which has an underlying micro-structure. The micro-structure makes the problem two-scale with a clear scale separation.
At each Gauss point of the macro-domain there exists a micro-simulation. The macro-domain is resolved in the file `macro-heat.py`
and the micro-domain is resolved in the file `micro_sim/micro_heat_circular.py`. Both the macro and micro problems are solved using the finite element library [Nutils](http://www.nutils.org/en/stable/).


The coupling between the macro-simulation and several micro-simulations is achieved using the coupling library [preCICE](https://precice.org/) 
and a Micro Manager. The Micro Manager (`micro-manager.py`) is a controlling components which handles all micro-simulations
and facilitates coupling with the macro-simulation via preCICE. The macro-problem and Micro Manager are configured via JSON files.

## Dependencies

* **Nutils** can be installed through the [installation procedure](http://www.nutils.org/en/latest/intro/#installation)
* **preCICE** can be installed through various ways which are described [here](https://precice.org/installation-overview.html)

## Running two-scale coupled heat conduction problem

The coupled macro problem can be started using the command:

```(python)
python3 macro-heat.py
```

For a coupled simulation the micro problems are managed by the micro manager and it is the micro manager which needs to be executed:

```(python)
python3 micro-manager.py
```

The micro manager can also be run in parallel in the following way:

```(python)
mpirun -n <num_procs> python3 micro-manager.py 
```

## How to configure a micro-simulation to be coupled via the Micro Manager

The micro-simulation script needs to be converted into a form having a class structure which would be callable from the Micro Manager.
The Micro Manager creates objects of this class for each micro-simulation and controls them till the end of the coupled simulation.
The Micro Manager script is intended to be used *as is*, and to facilitate that, the following naming conventions need to be followed:

* Create a class called `MicroSimulation` which consists of all the functions of the micro-simulation.
* Apart from the class constructor, define a function `initialize` which should consist of all steps to fully define the initial state of the micro-simulation
* Create a function named `solve` which should consist of all the solving steps for one time step of a micro-simulation or is the micro-problem is steady-state then solving until the steady-state is achieved.  The `solve` function will have all the steps which need to be done repeatedly.
* If implicit coupling is required between the macro- and micro- problems, then you can additionally define two functions `save_checkpoint` and `revert_to_checkpoint`.
  * `save_checkpoint` saves the state of the micro-problem such that if there is no convergence then the micro-problem can be reversed to this state.
  * `revert_to_checkpoint` reverts to the state which was saved earlier.
    
### Important points to consider when changing a micro-problem script

* Once the micro-problem script 

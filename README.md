# coupled-heat-conduction

<a style="text-decoration: none" href="https://github.com/precice/fenics-adapter/blob/master/LICENSE" target="_blank">
    <img src="https://img.shields.io/github/license/IshaanDesai/coupled-heat-conduction.svg" alt="GNU LGPL license">
</a>

This code solves a two-scale heat conduction model using the finite element library Nutils [1]. A micro scale model is coupled with a macro (Darcy) scale model using the coupling library preCICE [2]. This code is developed as part of the [SimTech PN5-9 project](https://www.simtech.uni-stuttgart.de/exc/research/pn/pn5/pn5-9/) at the University of Stuttgart. The macro code is `macro-heat-cond.py` and the micro code is `micro_sim/micro_heat_circular.py`. Both the codes can be run as single physics problems. For coupled problems the micro problems are managed via a `micro-manager.py` script. The macro problem and micro manager are configured via JSON files.

## Dependencies

* **Nutils** can be installed through the [installation procedure](http://www.nutils.org/en/latest/intro/#installation)
* **preCICE** can be installed through various ways which are described [here](https://precice.org/installation-overview.html)

## Running single physics problem

The macro scale code and the micro scale code can be run as single physics problems solving the heat equation for a conduction through solid. Single physics codes for both macro and micro scales can be run as follows:

```(python)
python3 macro-heat-cond.py
```

```(python)
python3 micro_heat_circular.py
```

## Running coupled problem

The coupled macro problem can be started using the command:

```(python)
python3 macro-heat-cond.py
```

For a coupled simulation the micro problems are managed by the micro manager and it is the micro manager which needs to be executed:

```(python)
python3 micro-manager.py
```

The micro manager can also be run in parallel in the following way:

```(python)
mpirun -n <num_procs> python3 micro-manager.py 
```

## Citing

[1] Gertjan van Zwieten, Joost van Zwieten, Clemens Verhoosel, Eivind Fonn, Timo van Opstal, & Wijnand Hoitinga. (2019). Nutils (5.0). Zenodo. https://doi.org/10.5281/zenodo.3243447

[2] H.-J. Bungartz, F. Lindner, B. Gatzhammer, M. Mehl, K. Scheufele, A. Shukaev, and B. Uekermann: preCICE - A Fully Parallel Library for Multi-Physics Surface Coupling. Computers and Fluids, 141, 250–258, 2016.

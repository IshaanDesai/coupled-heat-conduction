# coupled-heat-conduction

<a style="text-decoration: none" href="https://github.com/precice/fenics-adapter/blob/master/LICENSE" target="_blank">
    <img src="https://img.shields.io/github/license/IshaanDesai/coupled-heat-conduction.svg" alt="GNU LGPL license">
</a>

This code solves a two-scale heat conduction model using the finite element library Nutils [1]. A micro scale model having two materials is coupled with a macro scale model using the coupling library preCICE [2]. This code is developed as part of the [SimTech PN5-9 project](https://www.simtech.uni-stuttgart.de/exc/research/pn/pn5/pn5-9/) at the University of Stuttgart. The macro code is `macro-heat.py` and the micro code is `micro_sim/micro_heat_circular.py`. To couple multiple micro simulations to a single macro simulation, a managing component called *Micro Manager* is developed. The Micro Manager is in the file `micro-manager.py`. The macro problem and micro manager are configured via JSON files.

## Dependencies

* **Nutils** can be installed through the [installation procedure](http://www.nutils.org/en/latest/intro/#installation)
* **preCICE** can be installed through various ways which are described [here](https://precice.org/installation-overview.html)

## Running coupled problem

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

## Citing

[1] Gertjan van Zwieten, Joost van Zwieten, Clemens Verhoosel, Eivind Fonn, Timo van Opstal, & Wijnand Hoitinga. (2020). Nutils (6.2). Zenodo. https://doi.org/10.5281/zenodo.4071707

[2] H.-J. Bungartz, F. Lindner, B. Gatzhammer, M. Mehl, K. Scheufele, A. Shukaev, and B. Uekermann: preCICE - A Fully Parallel Library for Multi-Physics Surface Coupling. Computers and Fluids, 141, 250–258, 2016.

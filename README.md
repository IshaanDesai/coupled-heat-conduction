# coupled-heat-conduction

This code solves a two-scale heat conduction model using the finite element library Nutils. A micro grain scale model is coupled with a macro Darcy scale model using the coupling library preCICE. This code is developed as part of the [SimTech PN5-9 project](https://www.simtech.uni-stuttgart.de/exc/research/pn/pn5/pn5-9/) at the University of Stuttgart. The macro code is `macro-heat-cond.py` and the micro code is `micro-sims/micro-heat-cond-circular.py`. Both the codes can be run as single physics problems. For coupled problems the micro problems are managed via a `micro-manager.py` script. The macro problem and micro manager are configured via JSON files.

## Dependencies

* **Nutils** can be installed through the [installation procedure](http://www.nutils.org/en/latest/intro/#installation)
* **preCICE** can be installed through various ways which are described [here](https://precice.org/installation-overview.html)

## Running single physics problem

Both macro and micro scale codes can be run as single-physics problems solving the heat equation for a conduction through solid. Single physics codes for both macro and micro scales can be run as follows:

```(python)
python3 macro-heat-cond.py
```

```(python)
python3 micro-heat-cond-circular.py
```

## Running coupled problem

The coupled macro problem can be run as follows:

```(python)
python3 macro-heat-cond.py
```

For a coupled simulation the micro problems are managed by the micro-manager and it is the micro-manager which needs to be executed:

```(python)
python3 micro-manager.py
```

## Citing

[Nutils](https://zenodo.org/record/4071707)

preCICE is an academic project, developed at the [Technical University of Munich](https://www5.in.tum.de/) and at the [University of Stuttgart](https://www.ipvs.uni-stuttgart.de/): *H.-J. Bungartz, F. Lindner, B. Gatzhammer, M. Mehl, K. Scheufele, A. Shukaev, and B. Uekermann: preCICE - A Fully Parallel Library for Multi-Physics Surface Coupling. Computers and Fluids, 141, 250–258, 2016.*

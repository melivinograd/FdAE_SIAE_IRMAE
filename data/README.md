# Data Directory

All the data used for this paper was generated using [SPECTER](https://github.com/specter-cfd/SPECTER).

## Note on Datasets

The TFRecord datasets are not uploaded to this GitHub repository. However, they are available upon request.

## Available Files

1. `parameter_{Ra}.imp`: These files are used to simulate Rayleigh-Bénard flows for both Rayleigh numbers used in our experiments. The `parameter.imp` files are used with SPECTER, and instructions for their use can be found in the [SPECTER repository](https://github.com/specter-cfd/SPECTER). They should go in the `/bin` directory of SPECTER.

2. `makefile.in`: This file should be placed in the `/src` directory of the SPECTER repository to build the solver.

3. Initial conditions: The initial conditions used for the simulations are provided in files that start with `initial` and end in `.f90`.


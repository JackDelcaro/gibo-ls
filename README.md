# Code of the repo
- [optimizers](./src/optimizers.py): Implemented optimizers for black-box functions.
- [linear_search_routine](./src/linear_search_routine.py): Implemented linesearch routines.
- [model](./src/model.py): A Gaussian process model with a squared-exponential kernel that also supplies its Jacobian.
- [acquisition function](./src/acquisition_function.py): Custom acquisition function for gradient information.
- [loop](./src/loop.py): Brings together all parts necessary for an optimization loop.

# Installation
All required packages are specified in the conda yaml file: environment.yaml

## Synthetic Test Functions
### Run
Run python file simple_optimization_example.py to run the optimization on a simple 2D objective with unknown constraints. If the variable save_results is set to True the results are also saved in a pickle file in the data folder.

plot_animation.py shows the animation of the 2D example contained in saved_results.pkl (the file that is saved from simple_optimization_example.py)

plot_GP_example.py shows how the virtual and feasible GPs work for a 1D toy example
# Optimal-Execution-under-Incomplete-Information
Code for the paper 'optimal execution with partial information' (link: https://arxiv.org/abs/2411.04616). It includes numerical solvers and simulation experiments supporting the accompanying research paper.

## Repository Structure

## Core model and parameters
	•	parameters.py
Defines the global model parameters, grids, and numerical settings used throughout the project (time discretization, state grids, cost parameters, regime dynamics).

## Value function and neural network approximation
	•	Value_fun_NI.py
Main solver for the value function under incomplete information. Defines the neural network architecture used to parametrize the value function.
The neural network is used as an interpolation device within a fixed-point scheme, not as a direct policy optimizer.

	•	I0_no_func.py
Sample generator for the neural network.

	•	Operator_I1.py
Implements the fixed-point operator associated with execution decisions.


## Notebooks (analysis and visualization)
	•	Value_function_approx.ipynb
Numerical experiments illustrating the neural-network-based approximation of the value function and convergence of the fixed-point iterations.

	•	Value_function_exercice_region.ipynb
Visualization of continuation and execution regions.

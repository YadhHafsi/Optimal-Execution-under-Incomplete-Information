# Optimal-Execution-under-Incomplete-Information
Code for the paper 'optimal execution with partial information' (link: https://arxiv.org/abs/2411.04616). It includes numerical solvers for QVIs and simulation experiments supporting the accompanying research paper.

## Repository Structure

## Core model and parameters
	•	parameters.py
Defines the global model parameters, grids, and numerical settings used throughout the project (time discretization, state grids, cost parameters, regime dynamics).
	•	Prod.py
Implements core operators and helper routines used in the dynamic programming and QVI solvers.

## Value function and QVI solvers
	•	Value_fun_NI.py
Main solver for the value function under incomplete information.
Implements the dynamic programming recursion and the numerical solution of the QVI associated with the optimal execution problem.
	•	I0_no_func.py
Handles the continuation (no-intervention) operator corresponding to the inactive trading region.
	•	Operator_I1.py
Implements the intervention operator associated with execution decisions.

## Neural network approximation
	•	NN_parametrization.py
Defines the neural network architecture used to parametrize the value function.
The neural network is used as an interpolation device within a fixed-point scheme, not as a direct policy optimizer.

## Notebooks (analysis and visualization)
	•	Value_function_approx.ipynb
Numerical experiments illustrating the neural-network-based approximation of the value function and convergence of the fixed-point iterations.
	•	Value_function_exercice_region.ipynb
Visualization of continuation and execution regions implied by the solved QVI.

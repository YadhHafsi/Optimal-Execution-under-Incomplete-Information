from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from tqdm import tqdm
import numpy as np
import Value_fun_NI as res0
import I0_no_func as res1
import Operator_I1 as res2
import NN_parametrization as res3
import parameters as params

import torch

model = res1.I1_func_Net()  
model.load_state_dict(torch.load('i1net_model_sigmoid.pth'))
print('test')
# Use the model in the function
def I1_NN(t, u, y):
    nn_output = model(t, u, y)
    return nn_output.item()

def W1(t, y_2):
    # Generate grid points
    y = (y_2[0], y_2[2], y_2[3][0], np.array(y_2[4][0]))
    grid_points = np.linspace(t, T, 20)

    # Evaluate the function at grid points
    values = np.array([I1_NN(t, x, y) for x in grid_points])
    
    # Find the minimum value
    best_x_0 = np.max(values)
    return best_x_0


# Parallelized sample generation function
def generate_samples_I1(w, num_samples):
    num_workers = min(multiprocessing.cpu_count(), 1)  # Limit to 1 worker for Jupyter

    samples = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = [executor.submit(generate_single_sample, w) for _ in range(num_samples)]
        for future in tqdm(futures):
            try:
                result = future.result()
                samples.append(result)
            except Exception as e:
                print(f"Error occurred: {e}")

    return samples

# Worker function for parallelization
def generate_single_sample(w):
    try:
        # Call W1 with the loaded model
        t = np.random.uniform(0, T)
        u = np.random.uniform(t, T)
        x = np.random.uniform(0, 10)
        s = 100
        d = np.random.uniform(0, 10)
        kappa = np.random.rand(2) * 10
        kappa_0 = np.array([[kappa[0], kappa[1]], [xi * kappa[0], xi * kappa[1]]])

        mu = np.random.rand(1)[0]
        mu_0 = np.array([mu, 1 - mu])
        y_2 = (x, s, d, kappa_0, mu_0)
        best_value = W1(t, y_2)

        return best_value
    except Exception as e:
        print(f"Error in generate_single_sample: {e}")
        return None  # Handle error, returning None if sample generation fails


# Define a simple test function instead of lambda
def test_function(x):
    return 0

# Then call your generate_samples_I1 function like this
generate_samples_I1(test_function, 100)

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import Value_fun_NI as res0
import Operator_I1 as res2
import parameters as params
from tqdm import tqdm
from torch.cuda.amp import autocast, GradScaler
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from IPython.display import clear_output


#parameters
rho = params.rho
n_intgr = params.n_intgr
xi = params.xi
T = params.T

def I0_no_func(t, y):
    kappa, mu = y #x,s,d,
    return  res0.first_part_I0_ts(t, kappa, mu)
    #return  x*(s+d*np.exp(-rho*(T-t))) + res0.first_part_I0_ts(t, T, kappa, mu, 0, n_intgr)*x*nu_0 + res0.first_part_I0_ts(t, T, kappa, mu, rho, n_intgr)*x*(1-nu_0) - c_prime * x ** ((e + 1) / e) - c0

# Generate training samples
def generate_samples_I0(w, num_samples):
    samples = []
    for _ in range(num_samples):
        t = np.random.uniform(0, T)
        #x = np.random.uniform(20, 0)
        #s = np.random.uniform(30, 0)
        #d = np.random.uniform(15, 0)
        kappa = np.random.rand(2)*10
        kappa_0 = np.array([[kappa[0], kappa[1]], [kappa[1], kappa[0]]])

        mu = np.random.rand(1)[0]
        mu_0 = np.array([mu,1-mu])
        y_0 = (kappa_0,mu_0)#x, s, d,
        y = (kappa, mu)#x, s, d,
        target_value = w(t, y_0)
        samples.append((t, y, target_value))
    return samples

# Neural network definition
class I0_no_func_Net(nn.Module):
    def __init__(self):
        super(I0_no_func_Net, self).__init__()
        self.fc1 = nn.Linear(4, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, t, y):
        kappa, mu = y
        kappa_flat = kappa.flatten()
        mu_flat = mu.flatten()
        inputs = torch.tensor([t] + kappa_flat.tolist() + mu_flat.tolist(), dtype=torch.float32)#, x, s, d
        x = torch.sigmoid(self.fc1(inputs))
        x = torch.sigmoid(self.fc2(x))
        return self.fc3(x)

def Train_I0_no_func(w, sample_size = 50000, num_epochs = 50, load_model=False, path_name = 'i0net_model_sigmoid.pth'):
    # Instantiate the model
    model = I0_no_func_Net()  

    # Load the saved state dictionary
    if load_model:
        model.load_state_dict(torch.load(path_name))

    # Set the model to evaluation mode (optional, but recommended for inference)
    #model.eval()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop with error tracking
    train_samples = generate_samples_I0(w, sample_size)
    losses = []

    for epoch in tqdm(range(num_epochs)):
        epoch_loss = 0
        for t, y, target_value in train_samples:
            optimizer.zero_grad()
            outputs = model(t, y)
            target_tensor = torch.tensor([target_value], dtype=torch.float32)
            loss = criterion(outputs, target_tensor)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        # Use '\r' to overwrite the previous line and end='' to avoid adding a newline
        print(f'\rEpoch {epoch + 1}, Loss: {epoch_loss / len(train_samples):.6f}', flush = True)
        
        losses.append(epoch_loss / len(train_samples))
    return model, losses

# A function to generate a single sample
def generate_single_sample(w):
    t = np.random.uniform(0, T)
    u = np.random.uniform(t, T)
    x = np.random.uniform(0, 5)
    d = np.random.uniform(0, 5)
    kappa = np.random.rand(2) * 5
    kappa_0 = np.array([[kappa[0], kappa[1]], [kappa[1], kappa[0]]])

    mu = np.random.rand(1)[0]
    mu_0 = np.array([mu, 1 - mu])
    y_0 = (x, d, kappa_0, mu_0)
    y = (x, d, kappa, mu)

    # Compute the target value
    target_value = res2.I1(t, u, w, y_0)
    return (t, u, y, target_value)

# Parallelized sample generation function
def generate_samples_I1(w, num_samples):
    # Determine the number of CPU cores
    num_workers = multiprocessing.cpu_count()

    # Use ProcessPoolExecutor to parallelize the sample generation
    samples = []
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        # Create a list of futures to generate samples in parallel
        futures = [executor.submit(generate_single_sample, w) for _ in range(num_samples)]
        # Retrieve results as they complete
        for future in tqdm(futures):
            samples.append(future.result())

    return samples


def generate_samples_I1_old(w, num_samples):
    samples = []
    for _ in tqdm(range(num_samples)):
        t = np.random.uniform(0, T)
        u = np.random.uniform(t, T)
        x = np.random.uniform(0, 10)
        d = np.random.uniform(0, 15)
        kappa = np.random.rand(2)*10
        kappa_0 = np.array([[kappa[0], kappa[1]], [kappa[1], kappa[0]]])

        mu = np.random.rand(1)[0]
        mu_0 = np.array([mu,1-mu])
        y_0 = (x, d, kappa_0, mu_0)
        y = (x, d, kappa, mu)
        target_value = res2.I1(t, u, w, y_0)
        samples.append((t, u, y, target_value))
    return samples

# Neural network definition
class I1_func_Net(nn.Module):
    def __init__(self):
        super(I1_func_Net, self).__init__()
        self.fc1 = nn.Linear(7, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 16)
        self.fc4 = nn.Linear(16, 1)
    
    def forward(self, t, u, y):
        x, d, kappa, mu = y
        kappa_flat = kappa.flatten()
        mu_flat = mu.flatten()
        inputs = torch.tensor([t] + [u] + [x] + [d] + kappa_flat.tolist() + mu_flat.tolist(), dtype=torch.float32)
        x0 = torch.sigmoid(self.fc1(inputs))
        x0 = torch.sigmoid(self.fc2(x0))
        x0 = torch.sigmoid(self.fc3(x0))
        return self.fc4(x0)

def Train_I1_func(w, sample_size = 50000, num_epochs = 50, load_model=False, path_name = 'i0net_model_sigmoid.pth'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Instantiate the model and move it to the device (CPU or GPU)
    model = I1_func_Net().to(device)

    # Load the saved model if required
    if load_model:
        model.load_state_dict(torch.load(path_name))

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scaler = GradScaler()  # For mixed precision training

    # Generate training samples
    train_samples = generate_samples_I1(w, sample_size)

    # Convert train_samples into batches (here I use a simple batch size of 32 for demonstration)
    batch_size = 64
    num_batches = len(train_samples) // batch_size
    losses = []

    for epoch in tqdm(range(num_epochs), desc="Training Epochs"):
        epoch_loss = 0
        for i in range(num_batches):
            batch = train_samples[i * batch_size:(i + 1) * batch_size]

            # Separate batch into t, u, y, target_value and convert them to tensors
            t_batch = torch.tensor([x[0] for x in batch], dtype=torch.float32).to(device)
            u_batch = torch.tensor([x[1] for x in batch], dtype=torch.float32).to(device)
            y_batch = [(x[2][0], x[2][1], torch.tensor(x[2][2]), torch.tensor(x[2][3])) for x in batch]
            y_batch = [(torch.tensor(x[0]), torch.tensor(x[1]), x[2], x[3]) for x in y_batch]  # Convert nested y tuple
            target_value_batch = torch.tensor([x[3] for x in batch], dtype=torch.float32).to(device)

            optimizer.zero_grad()

            with autocast():  # Mixed precision training
                outputs = torch.stack([model(t, u, y) for t, u, y in zip(t_batch, u_batch, y_batch)])
                loss = criterion(outputs.squeeze(), target_value_batch)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()

        avg_epoch_loss = epoch_loss / num_batches
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_samples):.6f}')
        clear_output(wait=True)
        print(f'Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss / len(train_samples):.6f}')
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.6f}")
        losses.append(avg_epoch_loss)

    return model, losses
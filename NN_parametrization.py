import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import Value_fun_NI as res0
import parameters as params
from tqdm import tqdm

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
def generate_samples(w, num_samples):
    samples = []
    for _ in range(num_samples):
        t = np.random.uniform(0, T)
        #x = np.random.uniform(20, 0)
        #s = np.random.uniform(30, 0)
        #d = np.random.uniform(15, 0)
        kappa = np.random.rand(2)*10
        kappa_0 = np.array([[kappa[0], kappa[1]], [xi*kappa[0], xi*kappa[1]]])

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
    train_samples = generate_samples(w, sample_size, T)
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

# Generate training samples
def generate_samples_I1(w, num_samples):
    samples = []
    for _ in tqdm(range(num_samples)):
        t = np.random.uniform(0, T)
        x = np.random.uniform(20, 0)
        s = params.fund_price
        d = np.random.uniform(15, 0)
        kappa = np.random.rand(2)*10
        kappa_0 = np.array([[kappa[0], kappa[1]], [xi*kappa[0], xi*kappa[1]]])

        mu = np.random.rand(1)[0]
        mu_0 = np.array([mu,1-mu])
        y_0 = (x, s, d, kappa_0, mu_0)
        y = (x, d, kappa, mu)
        target_value = w(t, y_0)
        samples.append((t, y, target_value))
    return samples

# Neural network definition
class I1_func_Net(nn.Module):
    def __init__(self):
        super(I1_func_Net, self).__init__()
        self.fc1 = nn.Linear(6, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
    
    def forward(self, t, y):
        x, d, kappa, mu = y
        kappa_flat = kappa.flatten()
        mu_flat = mu.flatten()
        inputs = torch.tensor([t] + [x] + [d] + kappa_flat.tolist() + mu_flat.tolist(), dtype=torch.float32)
        x0 = torch.sigmoid(self.fc1(inputs))
        x0 = torch.sigmoid(self.fc2(x0))
        return self.fc3(x0)

def Train_I1_func(w, sample_size = 50000, num_epochs = 50, load_model=False, path_name = 'i0net_model_sigmoid.pth'):
    # Instantiate the model
    model = I1_func_Net()  

    # Load the saved state dictionary
    if load_model:
        model.load_state_dict(torch.load(path_name))

    # Set the model to evaluation mode (optional, but recommended for inference)
    #model.eval()

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Training loop with error tracking
    train_samples = generate_samples_I1(w, sample_size, T)
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
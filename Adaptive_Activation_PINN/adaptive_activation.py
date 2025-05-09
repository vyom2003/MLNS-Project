import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import time
import json

# Constants
BETA = 70.0
L = 2 * np.pi
LAMBDA = 0.1
T_MAX = 1.0
NUM_EPOCHS = 100000 
OPTIMIZER = "adam"  

# Custom activation modules
class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class SwishActivation(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Activation Generator Network
class ActivationGenerator(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=20, num_activations=6):
        super(ActivationGenerator, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_activations)
        )
        
        # Initialize weights properly
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Generate weights and apply softmax to normalize
        weights = F.softmax(self.network(x), dim=1)
        return weights

# Adaptive Activation Function
class AdaptiveActivation(nn.Module):
    def __init__(self, generator):
        super(AdaptiveActivation, self).__init__()
        
        self.generator = generator
        
        # Define activation functions as proper modules
        self.activations = nn.ModuleList([
            nn.Tanh(),
            nn.ReLU(),
            nn.Sigmoid(),
            SinActivation(),
            SwishActivation(),
            nn.LeakyReLU(0.1)
        ])
    
    def forward(self, x, input_coords):
        # Get weights from generator network
        weights = self.generator(input_coords)
        
        # Apply weighted sum of activation functions
        # Expand weights to match x's shape for broadcasting
        weights_expanded = weights.unsqueeze(-1)
        
        result = torch.zeros_like(x)
        # Apply each activation with its weight
        for i, activation in enumerate(self.activations):
            result += weights_expanded[:, i] * activation(x)
            
        return result

# Enhanced PINN with Adaptive Activation
class AdaptivePINN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=50, output_dim=1, num_layers=4):
        super(AdaptivePINN, self).__init__()
        
        # Activation generator network
        self.activation_generator = ActivationGenerator(input_dim=input_dim)
        # Adaptive activation
        self.adaptive_activation = AdaptiveActivation(self.activation_generator)
        
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            self.hidden_layers.append(nn.Linear(hidden_dim, hidden_dim))
        self.output_layer = nn.Linear(hidden_dim, output_dim)        
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
        
    def forward(self, x):
        # Save original input for activation generator
        original_input = x
        
        # Apply network with adaptive activation
        features = self.input_layer(x)
        features = self.adaptive_activation(features, original_input)
        for layer in self.hidden_layers:
            features = layer(features)
            features = self.adaptive_activation(features, original_input)
        output = self.output_layer(features)
        return output
    
    def get_activation_weights(self, x):
        return self.activation_generator(x)

# Loss functions
def pde_residual(collocation_points, model):
    u_pred = model(collocation_points)
    grad_u = torch.autograd.grad(
        u_pred,
        collocation_points,
        grad_outputs=torch.ones_like(u_pred),
        create_graph=True
    )[0]
    u_x = grad_u[:, 0:1]  # derivative with respect to x
    u_t = grad_u[:, 1:2]  # derivative with respect to t

    # PDE residual: (u_t + beta*u_x)^2
    residual = u_t + BETA * u_x
    return torch.mean(residual**2)

def periodic_boundary_loss(model, device):
    t_samples = torch.linspace(0, 1, 100, device=device).unsqueeze(1)
    x_left = torch.zeros_like(t_samples)
    x_right = torch.full_like(t_samples, L)

    left_input = torch.cat((x_left, t_samples), dim=1)
    right_input = torch.cat((x_right, t_samples), dim=1)

    u_left = model(left_input)
    u_right = model(right_input)
    # Enforce periodicity: u(0, t) ≈ u(2π, t)
    return torch.mean((u_left - u_right) ** 2)

def initial_condition_loss(model, device):
    x_samples = torch.linspace(0, L, 100, device=device).unsqueeze(1)
    t_zero = torch.zeros_like(x_samples)  # t = 0

    init_input = torch.cat((x_samples, t_zero), dim=1)
    u_init_pred = model(init_input)

    # True initial condition: u(x, 0) = sin(x)
    u_init_true = torch.sin(x_samples)

    return torch.mean((u_init_pred - u_init_true) ** 2)

def total_loss(model, collocation_points, device):
    loss_pde = pde_residual(collocation_points, model)
    loss_pb = periodic_boundary_loss(model, device)
    loss_ic = initial_condition_loss(model, device)
    return LAMBDA * loss_pde + loss_pb + loss_ic

# Training function with Adam optimizer
def train_model_adam(model, optimizer, collocation_points, device, num_epochs=NUM_EPOCHS):
    losses = []
    start_time = time.time()
    pbar = tqdm(range(num_epochs), desc="Training Epoch")
    for epoch in pbar:
        optimizer.zero_grad()
        loss = total_loss(model, collocation_points, device)
        loss.backward()
        optimizer.step()
        current_loss = loss.item()
        losses.append(current_loss)
        pbar.set_description(f"Loss: {current_loss:.6f}")
    
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    
    return losses, training_time

def setup_and_train():
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    # Load data or generate points
    try:
        data = pd.read_csv("collocation_points.csv")
        collocation_points = torch.tensor(data.values, dtype=torch.float32, requires_grad=True).to(device)
    except FileNotFoundError:
        print("Collocation points file not found, generating random points...")
        num_points = 2500
        x_points = torch.rand(num_points, 1) * L
        t_points = torch.rand(num_points, 1) * T_MAX
        collocation_points = torch.cat([x_points, t_points], dim=1).requires_grad_(True).to(device)
        
        # Save points to CSV for future use
        pd.DataFrame(collocation_points.detach().cpu().numpy()).to_csv("collocation_points.csv", index=False)

    # Initialize model
    model = AdaptivePINN().to(device)
    
    # Optimizer is always Adam
    print(f"Using Adam optimizer for {NUM_EPOCHS} epochs")
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    losses, training_time = train_model_adam(model, optimizer, collocation_points, device, NUM_EPOCHS)
    
    print(f"Training completed in {training_time:.2f} seconds")
    
    return model, losses, device, training_time

# Evaluation functions
def exact_solution_fft(x_array, t, h_values, beta, L):
    """
    Computes the exact solution of the 1D convection PDE using the Fourier transform approach.
    """
    N = len(x_array)
    H_k = np.fft.fft(h_values)
    k_array = 2 * np.pi * np.fft.fftfreq(N, d=(L / N))
    phase_factor = np.exp(-1j * beta * k_array * t)
    U_k = H_k * phase_factor
    u_vals = np.fft.ifft(U_k)
    return u_vals.real

def compute_solutions(model, device):
    Nx = 256
    Nt = 100
    x_vals = np.linspace(0, L, Nx, endpoint=False)
    t_vals = np.linspace(0, T_MAX, Nt)
    X, T = np.meshgrid(x_vals, t_vals)

    # Compute exact solution
    h_values = np.sin(x_vals)
    U_exact = np.zeros_like(X)
    for i, t_val in enumerate(t_vals):
        U_exact[i, :] = exact_solution_fft(x_vals, t_val, h_values, BETA, L)

    # Compute PINN solution
    XT = np.stack([X.ravel(), T.ravel()], axis=1)
    XT_tensor = torch.tensor(XT, dtype=torch.float32).to(device)
    
    model.eval()
    with torch.no_grad():
        U_pred_tensor = model(XT_tensor)
    
    U_pred = U_pred_tensor.cpu().numpy().reshape(Nt, Nx)
    U_diff = U_pred - U_exact
    
    return x_vals, t_vals, U_exact, U_pred, U_diff, XT_tensor

# Visualization functions
def plot_loss(losses):
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title(f"Training Loss with {OPTIMIZER.upper()} Optimizer (Adaptive)") # OPTIMIZER will be "adam"
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    # More fine-grained y-axis - use log scale with more ticks
    plt.yscale('log')
    plt.grid(True, which="both", ls="-")
    plt.minorticks_on()
    
    # Add more tick marks
    ax = plt.gca()
    
    # Customize the major and minor tick locations
    import matplotlib.ticker as ticker
    ax.yaxis.set_major_locator(ticker.LogLocator(base=10, numticks=15))
    ax.yaxis.set_minor_locator(ticker.LogLocator(base=10, subs=np.arange(0.1, 1.0, 0.1), numticks=15))
    
    plt.savefig(f"training_loss_{OPTIMIZER}_adaptive.png", dpi=300)
    plt.close()

def plot_solutions(x_vals, t_vals, U_exact, U_pred, U_diff):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    # (a) Exact solution
    c1 = axes[0].imshow(
        U_exact.T,
        extent=[0, T_MAX, 0, L],
        origin="lower",
        aspect="auto",
        cmap="rainbow"
    )
    axes[0].set_title(f"(a) Exact solution for β={BETA}")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("x")
    fig.colorbar(c1, ax=axes[0])

    # (b) PINN solution
    c2 = axes[1].imshow(
        U_pred.T,
        extent=[0, T_MAX, 0, L],
        origin="lower",
        aspect='auto',
        cmap='rainbow'
    )
    axes[1].set_title(f"(b) Adaptive PINN ({OPTIMIZER.upper()})")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("x")
    fig.colorbar(c2, ax=axes[1])

    # (c) Difference
    c3 = axes[2].imshow(
        U_diff.T,
        extent=[0, T_MAX, 0, L],
        origin="lower",
        aspect='auto',
        cmap='gray'
    )
    axes[2].set_title(f"(c) Difference (PINN - Exact)")
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("x")
    fig.colorbar(c3, ax=axes[2])

    plt.tight_layout()
    plt.savefig(f"solutions_{OPTIMIZER}_adaptive.png", dpi=300) # OPTIMIZER will be "adam"
    plt.close()

def plot_activation_weights(model, XT_tensor):
    # Get activation weights
    with torch.no_grad():
        weights = model.get_activation_weights(XT_tensor).cpu().numpy()
    
    # Define activation names
    activation_names = ['Tanh', 'ReLU', 'Sigmoid', 'Sin', 'Swish', 'LeakyReLU']
    
    # Reshape to grid
    Nt, Nx = 100, 256
    
    # Plot weights across the domain
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, name in enumerate(activation_names):
        weight_map = weights[:, i].reshape(Nt, Nx)
        im = axes[i].imshow(
            weight_map.T,
            extent=[0, T_MAX, 0, L],
            origin="lower",
            aspect="auto",
            cmap="viridis",
            vmin=0, vmax=1
        )
        axes[i].set_title(f"Weights for {name}")
        axes[i].set_xlabel("t")
        axes[i].set_ylabel("x")
        fig.colorbar(im, ax=axes[i])
    
    plt.tight_layout()
    plt.savefig(f"activation_weights_{OPTIMIZER}.png", dpi=300) # OPTIMIZER will be "adam"
    plt.close()

# Save results to file
def save_results(losses, metrics, training_time):
    # Create a dictionary with results
    results = {
        "model": "adaptive",
        "optimizer": OPTIMIZER,
        "beta": BETA,
        "num_epochs": NUM_EPOCHS,
        "training_time": training_time,
        "mse": metrics[0],
        "mae": metrics[1],
        "max_error": metrics[2],
        "final_loss": losses[-1] if losses else None
    }
    
    # Save metrics to JSON
    with open(f"results_{OPTIMIZER}_adaptive.json", "w") as f: # OPTIMIZER will be "adam"
        json.dump(results, f, indent=4)
    
    # Save losses array to CSV
    pd.DataFrame({"loss": losses}).to_csv(f"losses_{OPTIMIZER}_adaptive.csv", index=False) # OPTIMIZER will be "adam"
    
    print(f"Results saved to results_{OPTIMIZER}_adaptive.json and losses_{OPTIMIZER}_adaptive.csv")

# Analysis function
def compute_error_metrics(U_exact, U_pred):
    # Check for NaN values
    if np.isnan(U_pred).any():
        print("Warning: NaN values detected in prediction. Replacing with zeros.")
        U_pred = np.nan_to_num(U_pred)
    mse = np.mean((U_exact - U_pred)**2)
    mae = np.mean(np.abs(U_exact - U_pred))
    max_error = np.max(np.abs(U_exact - U_pred))
    
    print(f"MSE: {mse:.6e}")
    print(f"MAE: {mae:.6e}")
    print(f"Max Error: {max_error:.6e}")
    
    return mse, mae, max_error

# Main execution function
def main():
    model, losses, device, training_time = setup_and_train()
    plot_loss(losses)
    x_vals, t_vals, U_exact, U_pred, U_diff, XT_tensor = compute_solutions(model, device)
    plot_solutions(x_vals, t_vals, U_exact, U_pred, U_diff)
    plot_activation_weights(model, XT_tensor)
    metrics = compute_error_metrics(U_exact, U_pred)
    save_results(losses, metrics, training_time)
    torch.save(model.state_dict(), f"model_{OPTIMIZER}_adaptive.pth") # OPTIMIZER will be "adam"
    print(f"Model saved to model_{OPTIMIZER}_adaptive.pth")

if __name__ == "__main__":
    print(f"Adaptive Activation with {OPTIMIZER.upper()} Optimizer - Beta={BETA}") # OPTIMIZER will be "adam"
    main()
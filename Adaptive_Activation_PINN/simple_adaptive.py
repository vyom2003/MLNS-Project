import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
import pandas as pd
import os
import time
import json

# Constants
BETA = 70.0
L = 2 * np.pi
LAMBDA = 0.1
T_MAX = 1.0
NUM_EPOCHS = 100000  
OPTIMIZER = "adam"  

def log_progress(message):
    print(f"[INFO] {message}")

# Simple activation modules
class SinActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class SwishActivation(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Basic PINN model
class SimplePINN(nn.Module):
    def __init__(self, input_dim=2, hidden_dim=50, output_dim=1, num_layers=4):
        super(SimplePINN, self).__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.hidden_layers = nn.ModuleList()
        for _ in range(num_layers - 1):
            layer = nn.Linear(hidden_dim, hidden_dim)
            self.hidden_layers.append(layer)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self._init_weights()
        self.activations = nn.ModuleList([
            nn.Tanh(),
            nn.ReLU(),
            SinActivation(),
            SwishActivation()
        ])
        
        # Activation weights generator
        self.activation_weights = nn.Linear(input_dim, len(self.activations))
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Normalize input
        x_normalized = x.clone()
        x_normalized[:, 0] = x_normalized[:, 0] / L
        x_normalized[:, 1] = x_normalized[:, 1] / T_MAX
        
        # Get activation weights
        activation_weights = F.softmax(self.activation_weights(x_normalized), dim=1)
        
        features = self.input_layer(x_normalized)
        activated = torch.zeros_like(features)
        for i, activation in enumerate(self.activations):
            activated += activation_weights[:, i].unsqueeze(-1) * activation(features)
        
        features = activated
        
        for layer in self.hidden_layers:
            features = layer(features)
            
            # Apply weighted activations
            activated = torch.zeros_like(features)
            for i, activation in enumerate(self.activations):
                activated += activation_weights[:, i].unsqueeze(-1) * activation(features)
            
            features = activated
        output = self.output_layer(features)
        return output

# Basic loss functions
def pde_loss(model, x, t, device):
    """PDE: u_t + beta * u_x = 0"""
    points = torch.cat([x, t], dim=1).to(device)
    points.requires_grad_(True)
    
    u = model(points)
    
    # Compute gradients
    du_dx = torch.autograd.grad(
        u, points, 
        grad_outputs=torch.ones_like(u),
        create_graph=True, 
        retain_graph=True
    )[0]
    
    u_x = du_dx[:, 0:1]
    u_t = du_dx[:, 1:2]
    
    # PDE residual
    residual = u_t + BETA * u_x
    return torch.mean(residual**2)

def boundary_loss(model, device):
    """Periodic boundary: u(0,t) = u(2π,t)"""
    t = torch.linspace(0, T_MAX, 100, device=device).unsqueeze(1)
    x_left = torch.zeros_like(t)
    x_right = torch.full_like(t, L)
    
    points_left = torch.cat([x_left, t], dim=1)
    points_right = torch.cat([x_right, t], dim=1)
    
    u_left = model(points_left)
    u_right = model(points_right)
    
    return torch.mean((u_left - u_right)**2)

def initial_loss(model, device):
    """Initial condition: u(x,0) = sin(x)"""
    x = torch.linspace(0, L, 100, device=device).unsqueeze(1)
    t = torch.zeros_like(x)
    
    points = torch.cat([x, t], dim=1)
    
    u_pred = model(points)
    u_true = torch.sin(x)
    
    return torch.mean((u_pred - u_true)**2)

def total_loss(model, collocation_points, device):
    """Combined loss function"""
    # Split the collocation points into x and t
    x = collocation_points[:, 0:1]
    t = collocation_points[:, 1:2]
    
    loss_pde = pde_loss(model, x, t, device)
    loss_bc = boundary_loss(model, device)
    loss_ic = initial_loss(model, device)
    
    return LAMBDA * loss_pde + loss_bc + loss_ic

# Get collocation points
def get_collocation_points(device, num_points=2500):
    """Load collocation points from CSV or generate new ones"""
    csv_path = "collocation_points.csv"
    
    if os.path.exists(csv_path):
        log_progress(f"Loading collocation points from {csv_path}")
        data = pd.read_csv(csv_path)
        collocation_points = torch.tensor(data.values, dtype=torch.float32, requires_grad=True).to(device)
        log_progress(f"Loaded {len(collocation_points)} collocation points")
    else:
        log_progress(f"Generating {num_points} new collocation points")
        x_points = torch.rand(num_points, 1, device=device) * L
        t_points = torch.rand(num_points, 1, device=device) * T_MAX
        collocation_points = torch.cat([x_points, t_points], dim=1).requires_grad_(True)
        
        # Save to CSV for future use
        df = pd.DataFrame(collocation_points.detach().cpu().numpy())
        df.to_csv(csv_path, index=False)
        log_progress(f"Saved collocation points to {csv_path}")
    
    return collocation_points

# Training function with Adam optimizer
def train_adam(model, optimizer, device, collocation_points, num_epochs=NUM_EPOCHS):
    """Training function optimized for Adam"""
    losses = []
    start_time = time.time()
    pbar = tqdm(range(num_epochs))
    
    for epoch in pbar:
        optimizer.zero_grad()
        loss = total_loss(model, collocation_points, device)
        loss.backward()
        optimizer.step()
        loss_val = loss.item()
        losses.append(loss_val)
        pbar.set_description(f"Loss: {loss_val:.6f}")
        
    training_time = time.time() - start_time
    log_progress(f"Training completed in {training_time:.2f} seconds")    
    return losses, training_time

# Exact solution using FFT
def exact_solution(x_vals, t_vals):
    """Get exact solution using FFT"""
    Nx = len(x_vals)
    h_values = np.sin(x_vals)
    
    U_exact = np.zeros((len(t_vals), Nx))
    
    for i, t in enumerate(t_vals):
        H_k = np.fft.fft(h_values)
        k_array = 2 * np.pi * np.fft.fftfreq(Nx, d=(L/Nx))
        phase = np.exp(-1j * BETA * k_array * t)
        U_k = H_k * phase
        U_exact[i, :] = np.fft.ifft(U_k).real
    
    return U_exact

# Visualization
def plot_loss(losses):
    """Plot training loss"""
    plt.figure(figsize=(10, 6))
    plt.plot(losses)
    plt.title(f"Training Loss with {OPTIMIZER.upper()} Optimizer")
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
    
    plt.savefig(f"training_loss_simple_{OPTIMIZER}.png", dpi=300)
    plt.close()

def plot_solution(model, device):
    """Plot solution comparison"""
    # Grid for visualization
    Nx, Nt = 100, 50
    x_vals = np.linspace(0, L, Nx)
    t_vals = np.linspace(0, T_MAX, Nt)
    X, T = np.meshgrid(x_vals, t_vals)
    
    # Get exact solution
    U_exact = exact_solution(x_vals, t_vals)
    
    # Get PINN solution
    points = torch.tensor(np.column_stack((X.flatten(), T.flatten())), 
                         dtype=torch.float32, device=device)
    
    model.eval()
    with torch.no_grad():
        U_pred = model(points).cpu().numpy().reshape(Nt, Nx)
    
    # Compute difference
    U_diff = U_pred - U_exact
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Exact
    im1 = axes[0].imshow(U_exact.T, extent=[0, T_MAX, 0, L], 
                        origin='lower', aspect='auto', cmap='viridis')
    axes[0].set_title(f"Exact Solution (β={BETA})")
    axes[0].set_xlabel("t")
    axes[0].set_ylabel("x")
    fig.colorbar(im1, ax=axes[0])
    
    # PINN
    im2 = axes[1].imshow(U_pred.T, extent=[0, T_MAX, 0, L], 
                        origin='lower', aspect='auto', cmap='viridis')
    axes[1].set_title(f"Simple Adaptive PINN ({OPTIMIZER.upper()})")
    axes[1].set_xlabel("t")
    axes[1].set_ylabel("x")
    fig.colorbar(im2, ax=axes[1])
    
    # Difference
    im3 = axes[2].imshow(U_diff.T, extent=[0, T_MAX, 0, L], 
                        origin='lower', aspect='auto', cmap='coolwarm')
    axes[2].set_title("Difference (PINN - Exact)")
    axes[2].set_xlabel("t")
    axes[2].set_ylabel("x")
    fig.colorbar(im3, ax=axes[2])
    
    plt.tight_layout()
    plt.savefig(f"solutions_simple_{OPTIMIZER}.png", dpi=300)
    plt.close()
    
    # Error metrics
    mse = np.mean((U_exact - U_pred)**2)
    mae = np.mean(np.abs(U_exact - U_pred))
    max_error = np.max(np.abs(U_exact - U_pred))
    
    log_progress(f"MSE: {mse:.6e}")
    log_progress(f"MAE: {mae:.6e}")
    log_progress(f"Max Error: {max_error:.6e}")
    
    return mse, mae, max_error

# Save results to file
def save_results(losses, metrics, training_time):
    # Create a dictionary with results
    results = {
        "model": "simple",
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
    with open(f"results_simple_{OPTIMIZER}.json", "w") as f:
        json.dump(results, f, indent=4)
    
    # Save losses array to CSV
    pd.DataFrame({"loss": losses}).to_csv(f"losses_simple_{OPTIMIZER}.csv", index=False)
    
    log_progress(f"Results saved to results_simple_{OPTIMIZER}.json and losses_simple_{OPTIMIZER}.csv")

# Main function
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log_progress(f"Using device: {device}")
    model = SimplePINN().to(device)
    log_progress("Model created")
    collocation_points = get_collocation_points(device)
    log_progress(f"Using {len(collocation_points)} collocation points")
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    log_progress(f"Adam optimizer created, training for {NUM_EPOCHS} epochs")    
    losses, training_time = train_adam(model, optimizer, device, collocation_points)
    log_progress("Generating plots...")
    plot_loss(losses)
    metrics = plot_solution(model, device)
    save_results(losses, metrics, training_time)
    torch.save(model.state_dict(), f"model_simple_{OPTIMIZER}.pth")
    log_progress(f"Model saved to model_simple_{OPTIMIZER}.pth")
    log_progress("Done!")

if __name__ == "__main__":
    log_progress(f"Simple PINN with {OPTIMIZER.upper()} Optimizer - Beta={BETA}")
    main()
# Enhancing Physics-Informed Neural Networks through Region-Based Adaptation: Adaptive-Activation Functions and Weighted Loss Optimization

## MLNS 2025 Course Project
- Vyom Goyal (2021101099)
- Tanish Taneja (2021112011)


## Code Structure

```
.
├── Adaptive_Activation_PINN/
│   ├── adaptive_activation.py
│   ├── best_model.pth
│   ├── fixed_weights.py
│   └── simple_adaptive.py
├── Loss_Weighted_PINN/
│   ├── Kmean_pinn_grid_run.ipynb
│   ├── Loss_Kmean_pinn.ipynb
│   └── RAE_PINN.ipynb
├── baseLinePinn.ipynb
├── dataSampling.ipynb
└── README.md
```

## Key Components:

*   **`Adaptive_Activation_PINN/`**: Contains Python scripts related to PINNs with adaptive activation functions.
    *   `adaptive_activation.py`: Implements adaptive approach for finding activation function.
    *   `simple_adaptive.py`: Simpler version using just a layer in the PINN.
    *   `fixed_weights.py`: Baseline implementation with fixed weights of activation functions.
    *   `best_model.pth`: Saved model weights for best performing variation.
*   **`Loss_Weighted_PINN/`**: Contains Jupyter notebooks for PINNs with loss weighting strategies, using K-means clustering.
    *   `RAE_PINN.ipynb`: Notebook for implementing RAE PINN.
    *   `Loss_Kmean_pinn.ipynb`: Notebook exploring K-means for loss weighting in PINNs.
    *   `Kmean_pinn_grid_run.ipynb`: Notebook for running grid searches of multiple K-means PINN experiments.
*   **`baseLinePinn.ipynb`**: Jupyter notebook for a baseline PINN implementation.
*   **`dataSampling.ipynb`**: Jupyter notebook used for generating data for the PINN models.
*   **`README.md`**: This file.
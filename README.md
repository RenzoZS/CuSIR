---
lang: en-US
---

# CuSIR

## Introduction
CuSIR is a Python code built on top of CuPy, a NumPy-like library for GPU-accelerated computing. It provides a solver for the two-dimensional diffusive SIR model, described by the following system of reaction-diffusion equations:

$$
\begin{align}   
\partial_t S &= -\beta_{\mathbf{r}} S I - \gamma I + D_I \nabla^2 I \\ 
\partial_t I &= \beta_{\mathbf{r}} S I + D_S \nabla^2 S - \mathbf{v} \cdot \nabla I,
\end{align}
$${#system}
where $S$ is the density of susceptible individuals, $I$ is the density of infected individuals, $\beta_{\mathbf{r}}$ is the transmission rate that depends on the location $\mathbf{r}$, $\gamma$ is the recovery/removal rate, $D_I$ and $D_S$ are diffusion coefficients, and $\mathbf{v}$ is the convection field. 

## Implementation
The system is solved using a finite difference method on a uniform grid. The spatial domain is discretized into $L_x \times L_y$ cells. The system is updated using the following scheme:

$$
\begin{align}
S^{n+1}_{i,j} &= S^n_{i,j} - \beta_{\mathbf{r}} S^n_{i,j} I^n_{i,j} \Delta t - \gamma I^n_{i,j} \Delta t + D_I \left( \frac{S^n_{i+1,j} - 2 S^n_{i,j} + S^n_{i-1,j}}{\Delta x^2} + \frac{S^n_{i,j+1} - 2 S^n_{i,j} + S^n_{i,j-1}}{\Delta y^2} \right) \Delta t,\\
I^{n+1}_{i,j} &= I^n_{i,j} + \beta_{\mathbf{r}} S^n_{i,j} I^n_{i,j} \Delta t + D_S \left( \frac{I^n_{i+1,j} - 2 I^n_{i,j} + I^n_{i-1,j}}{\Delta x^2} + \frac{I^n_{i,j+1} - 2 I^n_{i,j} + I^n_{i,j-1}}{\Delta y^2} \right) \Delta t- v_{i,j} \left( \frac{I^n_{i+1,j} - I^n_{i-1,j}}{2 \Delta x}, \frac{I^n_{i,j+1} - I^n_{i,j-1}}{2 \Delta y} \right) \Delta t,
\end{align}
$$
where $\Delta t$ is the time step equals to $0.01$ by default, $\Delta x=1$ and $\Delta y=1$ are the spatial steps in the $x$ and $y$ directions, respectively, and $n$ is the time step index while $i,j$ are the spatial indices. The system is updated using a forward Euler scheme in time and a centered difference scheme in space. The system is solved with any given initial conditions and periodic or rigid boundary conditions in the $x$ and $y$ directions. The system is solved using a single GPU.


## Requirements

To use the CuSIR package, you will need the following software and hardware:

- A CUDA-compatible GPU: A graphics processing unit (GPU) that supports CUDA. Check the list of CUDA-compatible GPUs on the NVIDIA website (https://developer.nvidia.com/cuda-gpus) to see if your GPU is supported.

- CUDA Toolkit: A parallel computing platform and programming model developed by NVIDIA for general-purpose computing on GPUs. You can download CUDA from the NVIDIA website (https://developer.nvidia.com/cuda-downloads). 

*Note: Currently (January, 2023), the last version of CUDA (12) is not supported by CuPy. You will need to install any previous version of CUDA (recommended 11.2) to use CuSIR.* 

- CuPy: A NumPy-like library for GPU-accelerated computing. You can install CuPy by following the instructions in the CuPy documentation (https://docs-cupy.chainer.org/en/stable/install.html).

## Installation

To install the CuSIR package, you can use pip by running the following command in your command prompt or terminal:

```bash
pip install cusir
```

This command will install the latest version of the CuSIR package. It is mandatory to meet the requirements listed above for CuSIR to work properly.

## Usage

The CuSIR package provides a solver for the two-dimensional ([1-2](#heading-ids)) 
diffusive SIR system. The solver is implemented in the `system` class, which is located in the `system` module.

The following code shows how to use the `system` class to solve the diffusive SIR system:

```python
import cusir.system as cs

# Define the spatial domain
Lx = 2**10
Ly = 2**10

# Create the system object
s = cs.system(Lx, Ly)

# Define the system parameters
s.beta = 1 # Transmission rate
s.gamma = 0.1 # Recovery/removal rate  
s.D_I = 0.1 # Diffusion coefficient for infected individuals
s.D_S = 0.1 # Diffusion coefficient for susceptible individuals

# Define the initial conditions
s.set_plane_initial_conditions()

# Solve the system
for _ in range(10000):
    s.update() # Update the system
    s.rigid_x() # Apply rigid boundary conditions in the x-direction

#You can also use the following to do the same:
#s.solve(10000)

# Get the solution
S = s.S.get() # get() pulls the data from the GPU to the CPU as a NumPy array
I = s.I.get()
```




## License

CuSIR is licensed under the MIT license. See the `LICENSE` file for more details.





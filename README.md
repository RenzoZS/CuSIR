# CuSIR

## Introduction
CuSIR is a Python code built on top of CuPy, a NumPy-like library for GPU-accelerated computing. It provides a Cuda-based solver for the two-dimensional diffusive SIR model, described by the following system of reaction-diffusion equations:

$$
\begin{cases}
\partial_t S &= -\beta_{\mathbf{r}} S I - \gamma I + D_I \nabla^2 I,\\
\partial_t I &= \beta_{\mathbf{r}} S I + D_S \nabla^2 S,
\end{cases}
$$

where $S$ is the density of susceptible individuals, $I$ is the density of infected individuals, $\beta_{\mathbf{r}}$ is the transmission rate that depends on the location $\mathbf{r}$, $\gamma$ is the recovery/removal rate, $D_I$ and $D_S$ are diffusion coefficients, and $\nabla^2$ is the Laplacian operator. 


## Requirements

To use the CuSIR package, you will need the following software and hardware:

- A CUDA-compatible GPU: A graphics processing unit (GPU) that supports CUDA. Check the list of CUDA-compatible GPUs on the NVIDIA website (https://developer.nvidia.com/cuda-gpus) to see if your GPU is supported.

- CUDA: A parallel computing platform and programming model developed by NVIDIA for general-purpose computing on GPUs. You can download the latest version of CUDA from the NVIDIA website (https://developer.nvidia.com/cuda-downloads).

- CuPy: CuPy: A NumPy-like library for GPU-accelerated computing. You can install CuPy by following the instructions in the CuPy documentation (https://docs-cupy.chainer.org/en/stable/install.html).


Please note that the CuSIR package requires a GPU with compute capability 3.5 or higher. You can check the compute capability of your GPU by consulting the documentation for your GPU or by using a tool like nvidia-smi.

## Installation



## Usage


## Examples

## References

## License

## Contact:




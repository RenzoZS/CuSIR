# CuSIR

CuSIR is a Cuda based solver for a diffusive SIR model with spatial rate transmisión dependence. 

It basically solves the following system of equation with any given initial condition:

$$
\begin{align}
\partial_t S &= -\beta_{\mathbf{r}} S I - \gamma I + D_I \nabla^2 I,\\
\partial_t I &= \beta_{\mathbf{r}} S I + D_I \nabla^2 S.
\end{align}
$$




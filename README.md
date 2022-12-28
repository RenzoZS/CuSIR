# CuSIR

CuSIR is a Cuda based solver for the spatial SIR model. 

Basically solves the following system of equation with any given initial condition:

$$
\partial_t S = -\beta S I - \gamma I + D_I \nabla^2 I
$$
$$
\partial_t I = \beta S I + D_I \nabla^2 S.
$$



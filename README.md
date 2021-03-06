# Eigen34: Fast analytical Eigen-Decomposition of 3x3 and 4x4 Matrices
## Description
This repository contains code for analytical eigen-decomposition of 3x3 and 4x4 Matrices. Note that __this is a solver only for real eigenvalues___. Such matrices often turn-up in a variety of applications in computer graphics, vision and robotics. In these cases, analytical decomposition out-performs the standard numerical off-the-shelf algorithms and can prove very efficient in reducing overhead. Furthermore, they can be easily incorporate in a project since they are just two header files ( _EigenDecompose.h_ and _PolySolvers.h_ ).

## SVD
Although ___Singular Value Decomposition___ is not included as a function, but it can be easily obtained by simply eigen-decomposing the __Gram__ matrix (i.e., the product of the original matrix and its transposed). 

## Code examples
The respository contains sample code (_example.cpp_) for 3x3 and 4x4 matrix decompositions. Simple compile with a C++ flag:
```
g++ -std=c++14 example.cpp -o example
```

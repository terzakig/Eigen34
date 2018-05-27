# Eigen34
## Description
This repository contains code for analytical eigen-decomposition of 3x3 and 4x4 Matrices. Such matrices are met very often in a variety of applications in computer graphics, vision and robotics. In the case of such small-sized matrices, analytical decomposition out-performs the standard numerical off-the-shelf algorithms and can prove very efficient in reducing overhead. Furthermore, they can be easily incorporate in a project since they are just two header files ( ''EigenDecompose.h'' and ''PolySolvers.h'' )

## Code
The respoitory contains sample code (''example.cpp'') for 3x3 and 4x4 matrix decompositions. Simple compile with a C++ flag:
'''
g++ -std=c++14 example.cpp -o example
'''

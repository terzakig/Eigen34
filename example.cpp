/*
 * Examples of using the eigen34 namespace.
 *
 * George Terzakis 2018
 *
 */

#include <random>
#include <iostream>
#include <vector>
#include <cmath>

#include "PolySolvers.h"
#include "EigenDecompose.h"

using namespace std;

/**
 * @brief Verifies the eigendecomposition of an NxN matrix.
 * @tparam N The dimension of the matrix (e.g., 3 or 4).
 * @tparam P The precision type (float or double).
 * @param M Pointer to the flat NxN matrix (row-major).
 * @param decomp Pair containing eigenvalues and a vector of eigenvectors.
 * @return true if Av = λv for all pairs within epsilon.
 */
template <size_t N, typename P>
bool VerifyEigenDecomposition(const P *M, const std::pair<std::vector<P>, std::vector<std::vector<P>>>& decomp)
{
    const std::vector<P>& eigenvalues = decomp.first;
    const std::vector<std::vector<P>>& eigenvectors = decomp.second;

    // Adjust epsilon based on precision type
    const P epsilon = (sizeof(P) == 4) ? static_cast<P>(1e-4) : static_cast<P>(1e-8);

    for (size_t i = 0; i < decomp.first.size(); ++i) {
        const P lambda = eigenvalues[i];
        const std::vector<P>& v = eigenvectors[i];

        // 1. Compute Av and compare with λv in one pass
        for (size_t row = 0; row < N; ++row) {
            P Av_row = 0;
            for (size_t col = 0; col < N; ++col) {
                // Compile-time constant N ensures indexing is correct for 3x3 or 4x4
                Av_row += M[row * N + col] * v[col];
            }

            P lv_row = lambda * v[row];

            // 2. Check the difference
            if (std::abs(Av_row - lv_row) > epsilon) {
                std::cout << "Verification FAILED for " << N << "x" << N
                          << " matrix at Eigenpair " << i << ", Row "
                          << row << std::endl;
                return false;
            }
        }
    }

    std::cout << "Verification PASSED for " << N << "x" << N << " matrix." << std::endl;
    return true;
}

int main()
{

  // 1. Eigen-decomposition for a 3x3 matrix
  float A[] = {1, 5, 2,
               0, 1, 0,
               -1, 7, 11};

  auto decomposition3x3 = Eigen34::EigenDecompose3x3(A);

  cout << " Eigen-decomposition of a 3x3 matrix" << endl
       << " --------------------" << endl
       << endl;
  cout << "A = " << endl
       << "  [ " << A[0] << " , " << A[1] << " , " << A[2] << " ; " << endl
       << "    " << A[3] << " , " << A[4] << " , " << A[5] << " ; " << endl
       << "    " << A[6] << " , " << A[7] << " , " << A[8] << " ] " << endl
       << endl;
  // print the eigenvalues and respective eigenvectors
  for (size_t i = 0; i < decomposition3x3.first.size(); i++)
  {
    cout << " Eigenvalue : " << decomposition3x3.first[i] << " | Eigenvector : [ " << decomposition3x3.second[i][0] << " , " << decomposition3x3.second[i][1] << " , " << decomposition3x3.second[i][2] << " ]" << endl;
  }
  // verify the decomposition
  VerifyEigenDecomposition<3>(A, decomposition3x3);

  // 2. Eigen-decomposition for a 4x4 matrix
  float B[] = {1, 5, 2, -1,
               0, 1, 0, 8,
               -1, 7, 11, -2,
               1, -1, 3, 11};
  auto decomposition4x4 = Eigen34::EigenDecompose4x4(B);
  cout << endl
       << " Eigen-decomposition of a 4x4 matrix" << endl
       << " --------------------" << endl;
  cout << "B = " << endl
       << "  [ " << B[0] << " , " << B[1] << " , " << B[2] << " , " << B[3] << " ; " << endl
       << "    " << B[4] << " , " << B[5] << " , " << B[6] << " , " << B[7] << " ; " << endl
       << "    " << B[8] << " , " << B[9] << " , " << B[10] << " , " << B[11] << " ; " << endl
       << "    " << B[12] << " , " << B[13] << " , " << B[14] << " , " << B[15] << " ] " << endl
       << endl;

  // print the eigenvalues and respective eigenvectors
  for (size_t i = 0; i < decomposition4x4.first.size(); i++)
  {
    cout << " Eigenvalue : " << decomposition4x4.first[i]
         << " | Eigenvector : [ " << decomposition4x4.second[i][0] << " , " << decomposition4x4.second[i][1] << " , " << decomposition4x4.second[i][2]
         << " , " << decomposition4x4.second[i][3] << " ]" << endl;
  }
  // verify the decomposition
  VerifyEigenDecomposition<4>(B, decomposition4x4);

  // 3. Eigen-decomposition for a 4x4 positive semi-definite matrix
  //    This example also demonstrates matrix scaling
  float C[] = {0.64444, 0.12222, 0.53333, 1.25556,
               0.12222, 0.41111, 0.56667, 0.67778,
               0.53333, 0.56667, 1.00000, 1.56667,
               1.25556, 0.67778, 1.56667, 2.94444};
  cout << endl
       << " Eigen-decomposition of a 4x4 Positive Semidefinite Matrix" << endl
       << " ----------------------------------------------------" << endl;

  cout << "C = " << endl
       << "  [ " << C[0] << " , " << C[1] << " , " << C[2] << " , " << C[3] << " ; " << endl
       << "    " << C[4] << " , " << C[5] << " , " << C[6] << " , " << C[7] << " ; " << endl
       << "    " << C[8] << " , " << C[9] << " , " << C[10] << " , " << C[11] << " ; " << endl
       << "    " << C[12] << " , " << C[13] << " , " << C[14] << " , " << C[15] << " ] " << endl
       << endl;

  float frob_norm = 0.0;
  for (int i = 0; i < 16; i++)
    frob_norm += C[i] * C[i]; // Frobenius
  frob_norm = 1 / std::sqrt(frob_norm);
  // scale in nC, avoiding to destroy C
  float nC[sizeof(C) / sizeof(C[0])];
  for (int i = 0; i < 16; i++)
    nC[i] = C[i] * frob_norm;

  decomposition4x4 = Eigen34::EigenDecompose4x4(nC);
  for (auto& e : decomposition4x4.first)
    e /= frob_norm; // undo scaling: divide by inverse

  // print the eigenvalues and respective eigenvectors
  for (size_t i = 0; i < decomposition4x4.first.size(); i++)
  {
    cout << " Eigenvalue : " << decomposition4x4.first[i]
         << " | Eigenvector : [ " << decomposition4x4.second[i][0] << " , " << decomposition4x4.second[i][1] << " , " << decomposition4x4.second[i][2]
         << " , " << decomposition4x4.second[i][3] << " ]" << endl;
  }
  // verify the decomposition for the unscaled C
  VerifyEigenDecomposition<4>(C, decomposition4x4);

  return 1;
}

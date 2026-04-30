//
// Analytical Eigen-decomposition for 4x4 and 3x3 matrices
//
//
//   Robin Straebler - George Terzakis
//
//	 University of Portsmouth 2016
//

#ifndef __EIGENDECOMPOSE_H__
#define __EIGENDECOMPOSE_H__

#include <vector>
#include <utility>
#include <cmath>
#include <algorithm>

#include <iostream>
#include <ostream>
#include <cstdlib>

#include "PolySolvers.h"

// The 34 implies that the namespace contains algorithms
// for the eigen decomposition of 3x3 or 4x4 matrices.
//
namespace Eigen34
{
  template <typename P>
  std::vector<P> GaussJordan3x3(const std::vector<P>& flat_mat);

  template <typename P>
  std::vector<P> ComputeNullVector3x3(const std::vector<P>& flat_mat);

  namespace detail {
    // Compile-time choice of solver for 3x3 homogeneous systems (A-lam*I)*x = 0 :
    // true  -> row cross-products (null vector) method
    // false -> Gauss–Jordan elimination
    template <typename P, bool UseNullVectorMethod = true>
    std::vector<P> computeEigenVector3(const std::vector<P>& flat_mat)
    {
      if constexpr (UseNullVectorMethod)
        return ComputeNullVector3x3(flat_mat);
      else
        return GaussJordan3x3(flat_mat);
    }
  }


  /// <summary>
  /// This function performs two steps of Gauss-Jordan elimination in a 3x3 matrix.
  /// NOTE: The function assumes that the matrix provided is RANK-2 and therefore the solution
  //        will be the null space (vector). This way we obtain an eigen vector in two steps.
  /// </summary>
  /// <param name="std::vector<P> flat_mat"> is the coefficient matrix in flat form</param>
  /// <returns>std::vector<P> solutions </returns>
  template <typename P>
  std::vector<P> GaussJordan3x3(const std::vector<P>& flat_mat)
  {
    // For the first step of the Gauss-Jordan pivoting, we need the largest absolute value from flat_mat
    //
    auto absmax1it = std::max_element(flat_mat.begin(),
                                      flat_mat.end(),
                                      [](P a, P b)
                                      { return std::abs(a) < std::abs(b); });

    P max1 = *absmax1it;

    // Get the index of max (in absolute value)
    int index1 = absmax1it - flat_mat.begin();

    // Return empty vector if the maximum is zero (zero matrix)
    if (std::abs(max1) < PolySolvers::EPS<P>())
      return std::vector<P>();

    // For the second step of the Gauss-Jordan pivoting, we will require the
    //  maximum of the elements of the resulting matrix.
    // initializations silence warnings
    P max2 = 0;                // The value of the absolute max in the next stage of the elimination (only need 2 stages for 3x3 matrices)
    int index2 = 0 ;           // the index of the absolute maximum in the next stage of the elimination
    std::vector<P> flat_mat1;  // and the next matrix
    flat_mat1.reserve(4);

    // Variable which contain the coordinates of a vector
    P x1 = 0, x2 = 0, x3 = 0;

    // initializations silence warnings
    P *pX[3] = {&x1, &x2, &x3}; // pointer array to store permutations of x1, x2, x3
    P listFinalCoef[2] = {0, 0};

    // Taking cases according to where the maximum lies
    if (index1 == 0)
    {
      // List of the value after the first step of the Gauss-Jordan pivoting
      flat_mat1.push_back(flat_mat[4] - ((flat_mat[1] * flat_mat[3]) / max1));
      flat_mat1.push_back(flat_mat[5] - ((flat_mat[2] * flat_mat[3]) / max1));
      flat_mat1.push_back(flat_mat[7] - ((flat_mat[1] * flat_mat[6]) / max1));
      flat_mat1.push_back(flat_mat[8] - ((flat_mat[2] * flat_mat[6]) / max1));

      // Get the element with maximum absolute value and its index
      auto absmax2it = std::max_element(flat_mat1.begin(),
                                        flat_mat1.end(),
                                        [](P a, P b)
                                        { return std::abs(a) < std::abs(b); });
      max2 = *absmax2it;
      index2 = absmax2it - flat_mat1.begin();

      pX[0] = &x1;
      pX[1] = &x2;
      pX[2] = &x3;

      listFinalCoef[0] = flat_mat[1];
      listFinalCoef[1] = flat_mat[2];
    }
    else if (index1 == 1)
    {
      flat_mat1.push_back(flat_mat[3] - ((flat_mat[0] * flat_mat[4]) / max1));
      flat_mat1.push_back(flat_mat[5] - ((flat_mat[2] * flat_mat[4]) / max1));
      flat_mat1.push_back(flat_mat[6] - ((flat_mat[0] * flat_mat[7]) / max1));
      flat_mat1.push_back(flat_mat[8] - ((flat_mat[2] * flat_mat[7]) / max1));

      // Get the element with maximum absolute value and its index
      auto absmax2it = std::max_element(flat_mat1.begin(),
                                        flat_mat1.end(),
                                        [](P a, P b)
                                        { return std::abs(a) < std::abs(b); });
      max2 = *absmax2it;
      index2 = absmax2it - flat_mat1.begin();

      pX[0] = &x2;
      pX[1] = &x1;
      pX[2] = &x3;

      listFinalCoef[0] = flat_mat[0];
      listFinalCoef[1] = flat_mat[2];
    }
    else if (index1 == 2)
    {
      flat_mat1.push_back(flat_mat[3] - ((flat_mat[0] * flat_mat[5]) / max1));
      flat_mat1.push_back(flat_mat[4] - ((flat_mat[1] * flat_mat[5]) / max1));
      flat_mat1.push_back(flat_mat[6] - ((flat_mat[0] * flat_mat[8]) / max1));
      flat_mat1.push_back(flat_mat[7] - ((flat_mat[1] * flat_mat[8]) / max1));

      // Get the element with maximum absolute value and its index
      auto absmax2it = std::max_element(flat_mat1.begin(),
                                        flat_mat1.end(),
                                        [](P a, P b)
                                        { return std::abs(a) < std::abs(b); });
      max2 = *absmax2it;
      index2 = absmax2it - flat_mat1.begin();

      pX[0] = &x3;
      pX[1] = &x1;
      pX[2] = &x2;

      listFinalCoef[0] = flat_mat[0];
      listFinalCoef[1] = flat_mat[1];
    }
    else if (index1 == 3)
    {
      flat_mat1.push_back(flat_mat[1] - ((flat_mat[4] * flat_mat[0]) / max1));
      flat_mat1.push_back(flat_mat[2] - ((flat_mat[0] * flat_mat[5]) / max1));
      flat_mat1.push_back(flat_mat[7] - ((flat_mat[4] * flat_mat[6]) / max1));
      flat_mat1.push_back(flat_mat[8] - ((flat_mat[5] * flat_mat[6]) / max1));

      // Get the element with maximum absolute value and its index
      auto absmax2it = std::max_element(flat_mat1.begin(),
                                        flat_mat1.end(),
                                        [](P a, P b)
                                        { return std::abs(a) < std::abs(b); });
      max2 = *absmax2it;
      index2 = absmax2it - flat_mat1.begin();

      pX[0] = &x1;
      pX[1] = &x2;
      pX[2] = &x3;

      listFinalCoef[0] = flat_mat[4];
      listFinalCoef[1] = flat_mat[5];
    }
    else if (index1 == 4)
    {
      flat_mat1.push_back(flat_mat[0] - ((flat_mat[1] * flat_mat[3]) / max1));
      flat_mat1.push_back(flat_mat[2] - ((flat_mat[5] * flat_mat[1]) / max1));
      flat_mat1.push_back(flat_mat[6] - ((flat_mat[3] * flat_mat[7]) / max1));
      flat_mat1.push_back(flat_mat[8] - ((flat_mat[5] * flat_mat[7]) / max1));

      // Get the element with maximum absolute value and its index
      auto absmax2it = std::max_element(flat_mat1.begin(),
                                        flat_mat1.end(),
                                        [](P a, P b)
                                        { return std::abs(a) < std::abs(b); });
      max2 = *absmax2it;
      index2 = absmax2it - flat_mat1.begin();

      pX[0] = &x2;
      pX[1] = &x1;
      pX[2] = &x3;

      listFinalCoef[0] = flat_mat[3];
      listFinalCoef[1] = flat_mat[5];
    }
    else if (index1 == 5)
    {
      flat_mat1.push_back(flat_mat[0] - ((flat_mat[2] * flat_mat[3]) / max1));
      flat_mat1.push_back(flat_mat[1] - ((flat_mat[2] * flat_mat[4]) / max1));
      flat_mat1.push_back(flat_mat[6] - ((flat_mat[8] * flat_mat[3]) / max1));
      flat_mat1.push_back(flat_mat[7] - ((flat_mat[8] * flat_mat[4]) / max1));

      // Get the element with maximum absolute value and its index
      auto absmax2it = std::max_element(flat_mat1.begin(),
                                        flat_mat1.end(),
                                        [](P a, P b)
                                        { return std::abs(a) < std::abs(b); });
      max2 = *absmax2it;
      index2 = absmax2it - flat_mat1.begin();

      pX[0] = &x3;
      pX[1] = &x1;
      pX[2] = &x2;

      listFinalCoef[0] = flat_mat[3];
      listFinalCoef[1] = flat_mat[4];
    }
    else if (index1 == 6)
    {
      flat_mat1.push_back(flat_mat[1] - ((flat_mat[0] * flat_mat[7]) / max1));
      flat_mat1.push_back(flat_mat[2] - ((flat_mat[0] * flat_mat[8]) / max1));
      flat_mat1.push_back(flat_mat[4] - ((flat_mat[3] * flat_mat[7]) / max1));
      flat_mat1.push_back(flat_mat[5] - ((flat_mat[3] * flat_mat[8]) / max1));

      // Get the element with maximum absolute value and its index
      auto absmax2it = std::max_element(flat_mat1.begin(),
                                        flat_mat1.end(),
                                        [](P a, P b)
                                        { return std::abs(a) < std::abs(b); });
      max2 = *absmax2it;
      index2 = absmax2it - flat_mat1.begin();

      pX[0] = &x1;
      pX[1] = &x2;
      pX[2] = &x3;

      listFinalCoef[0] = flat_mat[7];
      listFinalCoef[1] = flat_mat[8];
    }
    else if (index1 == 7)
    {
      flat_mat1.push_back(flat_mat[0] - ((flat_mat[1] * flat_mat[6]) / max1));
      flat_mat1.push_back(flat_mat[2] - ((flat_mat[1] * flat_mat[8]) / max1));
      flat_mat1.push_back(flat_mat[3] - ((flat_mat[4] * flat_mat[6]) / max1));
      flat_mat1.push_back(flat_mat[5] - ((flat_mat[4] * flat_mat[8]) / max1));

      // Get the element with maximum absolute value and its index
      auto absmax2it = std::max_element(flat_mat1.begin(),
                                        flat_mat1.end(),
                                        [](P a, P b)
                                        { return std::abs(a) < std::abs(b); });
      max2 = *absmax2it;
      index2 = absmax2it - flat_mat1.begin();

      pX[0] = &x2;
      pX[1] = &x1;
      pX[2] = &x3;

      listFinalCoef[0] = flat_mat[6];
      listFinalCoef[1] = flat_mat[8];
    }
    else if (index1 == 8)
    {
      flat_mat1.push_back(flat_mat[0] - ((flat_mat[2] * flat_mat[6]) / max1));
      flat_mat1.push_back(flat_mat[1] - ((flat_mat[2] * flat_mat[7]) / max1));
      flat_mat1.push_back(flat_mat[3] - ((flat_mat[5] * flat_mat[6]) / max1));
      flat_mat1.push_back(flat_mat[4] - ((flat_mat[5] * flat_mat[7]) / max1));

      // Get the element with maximum absolute value and its index
      auto absmax2it = std::max_element(flat_mat1.begin(),
                                        flat_mat1.end(),
                                        [](P a, P b)
                                        { return std::abs(a) < std::abs(b); });
      max2 = *absmax2it;
      index2 = absmax2it - flat_mat1.begin();

      pX[0] = &x3;
      pX[1] = &x1;
      pX[2] = &x2;

      listFinalCoef[0] = flat_mat[6];
      listFinalCoef[1] = flat_mat[7];
    }

    if (std::abs(max2) != P(0))
    {
      if (index2 == 0)
      {
        *(pX[2]) = 1;
        *(pX[1]) = -flat_mat1[1] / max2;
      }
      else if (index2 == 1)
      {
        *(pX[2]) = -flat_mat1[0] / max2;
        *(pX[1]) = 1;
      }
      else if (index2 == 2)
      {
        *(pX[2]) = 1;
        *(pX[1]) = -flat_mat1[3] / max2;
      }
      else if (index2 == 3)
      {
        *(pX[2]) = -flat_mat1[2] / max2;
        *(pX[1]) = 1;
      }
    }
    else
    {
      // zero 2x2 sub-block, system is underdetermined
      // pick an arbitrary but valid vector in the null space

      *(pX[2]) = 0;
      *(pX[1]) = 1;
    }

    *(pX[0]) = -((listFinalCoef[0] * *(pX[1])) / max1) - ((listFinalCoef[1] * *(pX[2])) / max1);

    // normalize the solution
    P normSq = x1 * x1 + x2 * x2 + x3 * x3;
    if (normSq == P(0))
      return std::vector<P>(); // return empty vector
    P invnorm = P(1) / std::sqrt(normSq);

    return std::vector<P>({x1 * invnorm, x2 * invnorm, x3 * invnorm});
  }

  ///////////////////////////////////////////////////////////////////////////////////////////////
  /// Computes the eigenvector for a specific real eigenvalue lambda
  /// from the *characteristic* matrix M = A - lambda*I of a 3x3 matrix A.
  /// This is a cleaner alternative to GaussJordan3x3()
  ///
  /// Note: the function accepts matrix M, not A!
  ///       More importantly, it breaks in case of nearly multiple eigenvalues.
  ///
  /// Based on the observation that the rows of M = (A−lambda*I) are orthogonal to
  /// the eigenvector so the cross product of any two rows yields the eigenvector.
  ///
  /// Robustness comes from the fact that M is singular (rank <= 2). If we pick two
  /// nearly parallel rows, their cross product will be numerically zero.
  /// This is mitigated by calculating all possible row-pair cross products and choosing
  /// the largest magnitude one
  template <typename P>
  std::vector<P> ComputeNullVector3x3(const std::vector<P>& flat_mat)
  {
    // Rows as pointers to flat_mat
    const P *M = flat_mat.data();
    const P *row0 = M + 0;
    const P *row1 = M + 3;
    const P *row2 = M + 6;

    // Compute all possible cross products between rows
    auto crossprod = [](const P *a, const P *b, P *axb)
    {
      axb[0] = a[1] * b[2] - a[2] * b[1];
      axb[1] = a[2] * b[0] - a[0] * b[2];
      axb[2] = a[0] * b[1] - a[1] * b[0];
    };
    P v01[3], v12[3], v20[3];
    crossprod(row0, row1, v01);
    crossprod(row1, row2, v12);
    crossprod(row2, row0, v20);

    // Find which cross product has the largest magnitude
    // (This avoids issues with parallel or zero rows)
    P d01 = v01[0] * v01[0] + v01[1] * v01[1] + v01[2] * v01[2];
    P d12 = v12[0] * v12[0] + v12[1] * v12[1] + v12[2] * v12[2];
    P d20 = v20[0] * v20[0] + v20[1] * v20[1] + v20[2] * v20[2];

    P *res, normsq;
    if (d01 >= d12 && d01 >= d20) {
      res = v01;
      normsq = d01;
    } else if (d12 >= d01 && d12 >= d20) {
      res = v12;
      normsq = d12;
    } else {
      res = v20;
      normsq = d20;
    }

    std::vector<P> evec({res[0], res[1], res[2]});

    // Final check: If all cross products are zero, the matrix might
    // have rank 1 (multiple eigenvalues). In that case, we pick
    // a row and find any vector orthogonal to it.
    constexpr P eps_sq = PolySolvers::EPS<P>() * PolySolvers::EPS<P>();
    if (normsq < eps_sq) {
        // fallback for rank-1 cases (degenerate matrices)
        if (row0[0] * row0[0] + row0[1] * row0[1] + row0[2] * row0[2] > eps_sq) {
          if (std::abs(row0[0]) > std::abs(row0[2])) {
            evec[0] = -row0[1];
            evec[1] =  row0[0];
            evec[2] = 0;
          }
          else {
            evec[0] = 0;
            evec[1] = -row0[2];
            evec[2] =  row0[1];
          }
        } else {
          // absolute degenerate case
          evec[0] = 1;
          evec[1] = evec[2] = 0;
        }
    }

    // normalize
    P nrm = std::sqrt(evec[0] * evec[0] + evec[1] * evec[1] + evec[2] * evec[2]);
    if (nrm > PolySolvers::EPS<P>()) {
      nrm = 1 / nrm;
      evec[0] *= nrm;
      evec[1] *= nrm;
      evec[2] *= nrm;
    }

    return evec;
  }
  ///////////////////////////////////////////////////////////////////////////////////////////////

  ///
  /// This function eliminates the coefficients of the column "col" based on an element at (row, col)
  ///
  ///
  template <typename P>
  std::vector<P> GaussJordanFirstStep(const std::vector<P>& flat_mat, int row, int col)
  {
    std::vector<P> result;
    result.reserve(9); // 16-2*4+1

    for (int i = 0; i < 4; i++)
    {
      for (int j = 0; j < 4; j++)
      {

        if (i != row && j != col)
        {
          result.push_back(flat_mat[i * 4 + j] - ((flat_mat[row * 4 + j] * flat_mat[i * 4 + col]) / flat_mat[row * 4 + col]));
        }
      }
    }

    return result;
  }

  ///
  /// The Gauss-Jordan elimination for Rank-3 4x4 matrices
  /// The steps are fixed and lead to a solution up to arbitrary scale.
  /// This is how we solve for the eigenvectors of a 4x4 matrix.
  ///
  /// The input vector is a flat version of the 4x4 matrix
  ///
  template <typename P>
  std::vector<P> GaussJordan4x4(const std::vector<P>& flat_mat)
  {
    // Working out the index of the max coefficient (absolute value).
    auto absmax1it = std::max_element(flat_mat.begin(),
                                      flat_mat.end(),
                                      [](P a, P b)
                                      { return std::abs(a) < std::abs(b); });

    P max1 = *absmax1it;
    int index1 = absmax1it - flat_mat.begin();

    // Return empty vector if the maximum is zero (zero matrix)
    if (std::abs(max1) < PolySolvers::EPS<P>())
      return std::vector<P>();

    // Create new variable which is used later.
    std::vector<P> flat_mat1;

    P x1 = 0, x2 = 0, x3 = 0, x4 = 0;
    // Cache for the result of 3x3 Gauss Jordan elimination
    std::vector<P> reducedEigenVec;

    if (index1 == 0)
    {
      // We solve the reduced 3x3 system after eliminating column 0
      flat_mat1 = GaussJordanFirstStep(flat_mat, 0, 0);
      reducedEigenVec = detail::computeEigenVector3(flat_mat1);
      x2 = reducedEigenVec[0];
      x3 = reducedEigenVec[1];
      x4 = reducedEigenVec[2];
      x1 = -(flat_mat[1] * x2 + flat_mat[2] * x3 + flat_mat[3] * x4) / flat_mat[0];
    }
    else if (index1 == 1)
    {
      // We solve the reduced 3x3 system after eliminating column 1
      flat_mat1 = GaussJordanFirstStep(flat_mat, 0, 1);
      reducedEigenVec = detail::computeEigenVector3(flat_mat1);
      x1 = reducedEigenVec[0];
      x3 = reducedEigenVec[1];
      x4 = reducedEigenVec[2];
      x2 = -(flat_mat[0] * x1 + flat_mat[2] * x3 + flat_mat[3] * x4) / flat_mat[1];
    }
    else if (index1 == 2)
    {
      flat_mat1 = GaussJordanFirstStep(flat_mat, 0, 2);
      reducedEigenVec = detail::computeEigenVector3(flat_mat1);
      x1 = reducedEigenVec[0];
      x2 = reducedEigenVec[1];
      x4 = reducedEigenVec[2];
      x3 = -(flat_mat[0] * x1 + flat_mat[1] * x2 + flat_mat[3] * x4) / flat_mat[2];
    }
    else if (index1 == 3)
    {
      flat_mat1 = GaussJordanFirstStep(flat_mat, 0, 3);
      reducedEigenVec = detail::computeEigenVector3(flat_mat1);
      x1 = reducedEigenVec[0];
      x2 = reducedEigenVec[1];
      x3 = reducedEigenVec[2];
      x4 = -(flat_mat[0] * x1 + flat_mat[1] * x2 + flat_mat[2] * x3) / flat_mat[3];
    }
    else if (index1 == 4)
    {
      flat_mat1 = GaussJordanFirstStep(flat_mat, 1, 0);
      reducedEigenVec = detail::computeEigenVector3(flat_mat1);
      x2 = reducedEigenVec[0];
      x3 = reducedEigenVec[1];
      x4 = reducedEigenVec[2];
      x1 = -(flat_mat[5] * x2 + flat_mat[6] * x3 + flat_mat[7] * x4) / flat_mat[4];
    }
    else if (index1 == 5)
    {
      flat_mat1 = GaussJordanFirstStep(flat_mat, 1, 1);
      reducedEigenVec = detail::computeEigenVector3(flat_mat1);
      x1 = reducedEigenVec[0];
      x3 = reducedEigenVec[1];
      x4 = reducedEigenVec[2];
      x2 = -(flat_mat[4] * x1 + flat_mat[6] * x3 + flat_mat[7] * x4) / flat_mat[5];
    }
    else if (index1 == 6)
    {
      flat_mat1 = GaussJordanFirstStep(flat_mat, 1, 2);
      reducedEigenVec = detail::computeEigenVector3(flat_mat1);
      x1 = reducedEigenVec[0];
      x2 = reducedEigenVec[1];
      x4 = reducedEigenVec[2];
      x3 = -(flat_mat[4] * x1 + flat_mat[5] * x2 + flat_mat[7] * x4) / flat_mat[6];
    }
    else if (index1 == 7)
    {
      flat_mat1 = GaussJordanFirstStep(flat_mat, 1, 3);
      reducedEigenVec = detail::computeEigenVector3(flat_mat1);
      x1 = reducedEigenVec[0];
      x2 = reducedEigenVec[1];
      x3 = reducedEigenVec[2];
      x4 = -(flat_mat[4] * x1 + flat_mat[5] * x2 + flat_mat[6] * x3) / flat_mat[7];
    }
    else if (index1 == 8)
    {
      flat_mat1 = GaussJordanFirstStep(flat_mat, 2, 0);
      reducedEigenVec = detail::computeEigenVector3(flat_mat1);
      x2 = reducedEigenVec[0];
      x3 = reducedEigenVec[1];
      x4 = reducedEigenVec[2];
      x1 = -(flat_mat[9] * x2 + flat_mat[10] * x3 + flat_mat[11] * x4) / flat_mat[8];
    }
    else if (index1 == 9)
    {
      flat_mat1 = GaussJordanFirstStep(flat_mat, 2, 1);
      reducedEigenVec = detail::computeEigenVector3(flat_mat1);
      x1 = reducedEigenVec[0];
      x3 = reducedEigenVec[1];
      x4 = reducedEigenVec[2];
      x2 = -(flat_mat[8] * x1 + flat_mat[10] * x3 + flat_mat[11] * x4) / flat_mat[9];
    }
    else if (index1 == 10)
    {
      flat_mat1 = GaussJordanFirstStep(flat_mat, 2, 2);
      reducedEigenVec = detail::computeEigenVector3(flat_mat1);
      x1 = reducedEigenVec[0];
      x2 = reducedEigenVec[1];
      x4 = reducedEigenVec[2];
      x3 = -(flat_mat[8] * x1 + flat_mat[9] * x2 + flat_mat[11] * x4) / flat_mat[10];
    }
    else if (index1 == 11)
    {
      flat_mat1 = GaussJordanFirstStep(flat_mat, 2, 3);
      reducedEigenVec = detail::computeEigenVector3(flat_mat1);
      x1 = reducedEigenVec[0];
      x2 = reducedEigenVec[1];
      x3 = reducedEigenVec[2];
      x4 = -(flat_mat[8] * x1 + flat_mat[9] * x2 + flat_mat[10] * x3) / flat_mat[11];
    }
    else if (index1 == 12)
    {
      flat_mat1 = GaussJordanFirstStep(flat_mat, 3, 0);
      reducedEigenVec = detail::computeEigenVector3(flat_mat1);
      x2 = reducedEigenVec[0];
      x3 = reducedEigenVec[1];
      x4 = reducedEigenVec[2];
      x1 = -(flat_mat[13] * x2 + flat_mat[14] * x3 + flat_mat[15] * x4) / flat_mat[12];
    }
    else if (index1 == 13)
    {
      flat_mat1 = GaussJordanFirstStep(flat_mat, 3, 1);
      reducedEigenVec = detail::computeEigenVector3(flat_mat1);
      x1 = reducedEigenVec[0];
      x3 = reducedEigenVec[1];
      x4 = reducedEigenVec[2];
      x2 = -(flat_mat[12] * x1 + flat_mat[14] * x3 + flat_mat[15] * x4) / flat_mat[13];
    }
    else if (index1 == 14)
    {
      flat_mat1 = GaussJordanFirstStep(flat_mat, 3, 2);
      reducedEigenVec = detail::computeEigenVector3(flat_mat1);
      x1 = reducedEigenVec[0];
      x2 = reducedEigenVec[1];
      x4 = reducedEigenVec[2];
      x3 = -(flat_mat[12] * x1 + flat_mat[13] * x2 + flat_mat[15] * x4) / flat_mat[14];
    }
    else if (index1 == 15)
    {
      flat_mat1 = GaussJordanFirstStep(flat_mat, 3, 3);
      reducedEigenVec = detail::computeEigenVector3(flat_mat1);
      x1 = reducedEigenVec[0];
      x2 = reducedEigenVec[1];
      x3 = reducedEigenVec[2];
      x4 = -(flat_mat[12] * x1 + flat_mat[13] * x2 + flat_mat[14] * x3) / flat_mat[15];
    }

    // normalize the solution
    P normSq = x1 * x1 + x2 * x2 + x3 * x3 + x4 * x4;
    if (normSq == P(0))
      return std::vector<P>(); // return empty vector
    P invnorm = P(1) / std::sqrt(normSq);

    return std::vector<P>({x1 * invnorm, x2 * invnorm, x3 * invnorm, x4 * invnorm});
  }

  // sorting networks for sorting small vectors in ascending order
  template <typename P>
  inline static void swap_(P& a, P& b)
  {
    P temp = a;
    a = b;
    b = temp;
  }

  // sort a vector of size two with the network [[0 1]]
  template <typename P>
  inline static void sortnet2(std::vector<P>& vec)
  {
    if (vec[0] > vec[1]) swap_(vec[0], vec[1]);
  }

  // sort a vector of size three with the network [[0 1][0 2][1 2]]
  template <typename P>
  inline static void sortnet3(std::vector<P>& vec)
  {
    if (vec[0] > vec[1]) swap_(vec[0], vec[1]);
    if (vec[0] > vec[2]) swap_(vec[0], vec[2]);
    if (vec[1] > vec[2]) swap_(vec[1], vec[2]);
  }

  // sort a vector of size four with the network [[0 1][2 3][0 2][1 3][1 2]]
  template <typename P>
  inline static void sortnet4(std::vector<P>& vec)
  {
    if (vec[0] > vec[1]) swap_(vec[0], vec[1]);
    if (vec[2] > vec[3]) swap_(vec[2], vec[3]);
    if (vec[0] > vec[2]) swap_(vec[0], vec[2]);
    if (vec[1] > vec[3]) swap_(vec[1], vec[3]);
    if (vec[1] > vec[2]) swap_(vec[1], vec[2]);
  }

  // sort a vector of size up to four with the appropriate network
  template <typename P>
  inline static void sortnet(std::vector<P>& vec)
  {
    switch (vec.size())
    {
      case 0:
      case 1:
	      // nothing to do
	      return;
      case 2:
	      sortnet2(vec);
	      return;
      case 3:
	      sortnet3(vec);
	      return;
      case 4:
	      sortnet4(vec);
	      return;
      default:
	      std::cerr << "Internal error in Eigen34::sortnet(), got " << vec.size() << "!\n";
	      std::abort();
    }
  }

  ///
  /// Return the list of the eigenvalues of a 4x4 matrix.
  /// Argument is a 1D array representing the 4x4 matrix
  ///
  template <typename P>
  std::vector<P> EigenValues4x4(const P *array)
  {

    P a11 = array[0 * 4 + 0], a12 = array[0 * 4 + 1], a13 = array[0 * 4 + 2], a14 = array[0 * 4 + 3];
    P a21 = array[1 * 4 + 0], a22 = array[1 * 4 + 1], a23 = array[1 * 4 + 2], a24 = array[1 * 4 + 3];
    P a31 = array[2 * 4 + 0], a32 = array[2 * 4 + 1], a33 = array[2 * 4 + 2], a34 = array[2 * 4 + 3];
    P a41 = array[3 * 4 + 0], a42 = array[3 * 4 + 1], a43 = array[3 * 4 + 2], a44 = array[3 * 4 + 3];

    // 1. Obtaining the coefficients of the characteristic polynomial of A11 - λ*I where A11 is the (1, 1) submatrix of A:
    // A11 = [a22-λ a23 a24;
    //        a32 a33-λ a34;
    //        a42 a43 a44 - λ]

    // The coefficients of the cubic polynomial det(A11-λI) are:
    P C3 = -1, C2 = a22 + a33 + a44;
    P C1 = a42 * a24 + a32 * a23 + a34 * a43 - a22 * a33 - a22 * a44 - a33 * a44;
    P C0 = a22 * a33 * a44 + a23 * a42 * a34 + a24 * a32 * a43 - a22 * a34 * a43 - a23 * a32 * a44 - a24 * a42 * a33;

    // 2. Now multiplying with a11 to get a quartic polynomial with coefficients W4, W3, W2, W1, W0 as follows:
    P W4 = -C3,
      W3 = a11 * C3 - C2,
      W2 = a11 * C2 - C1,
      W1 = a11 * C1 - C0,
      W0 = a11 * C0;

    // 4. Now we obtain the coefficients of the 3 quadratics (from the algebraic complements A12, A13, A14) as follows:
    // a. (A-λI)12 (-)
    P Q1_2 = -a12 * a21,
      Q1_1 = -a12 * (-a21 * (a33 + a44) + a23 * a31 + a24 * a41),
      Q1_0 = -a12 * (a24 * (a31 * a43 - a41 * a33) - a23 * (a31 * a44 - a41 * a34) + a21 * (a33 * a44 - a43 * a34)); // good all being well...
                                                                                                                     // multiplying with -a12
                                                                                                                     // P1_2 *= -a12; P1_1 *= -a12; P1_0 *= -a12;

    // b. (A-λ)13 (+)
    P Q2_2 = a13 * -a31,
      Q2_1 = a13 * ((a31 * a44 - a41 * a34 + a22 * a31) - a21 * a32),
      Q2_0 = a13 * (a24 * (a31 * a42 - a41 * a32) + a22 * (a41 * a34 - a31 * a44) + a21 * (a32 * a44 - a42 * a34));

    // c. (A-λ)14 (-)
    P Q3_2 = -a14 * a41,
      Q3_1 = -a14 * ((a31 * a43 - a41 * a33 - a22 * a41) + a21 * a42),
      Q3_0 = -a14 * (a23 * (a31 * a42 - a41 * a32) - a22 * (a31 * a43 - a41 * a33) + a21 * (a32 * a43 - a42 * a33));

    // 5. Final coefficients
    P A0 = Q3_0 + Q2_0 + Q1_0 + W0,
      A1 = Q3_1 + Q2_1 + Q1_1 + W1,
      A2 = Q3_2 + Q2_2 + Q1_2 + W2,
      A3 = W3,
      A4 = W4;

    std::vector<P> solution = PolySolvers::SolveQuartic(A4, A3, A2, A1, A0);

    // Put the solutions in ascending order
    // std::sort(solution.begin(), solution.end());
    sortnet(solution);

    return solution;
  }

  ///
  /// Return the list of the eigenvalues of a 3x3 matrix.
  /// Argument is a 1D array representing the 3x3 matrix
  ///
  template <typename P>
  std::vector<P> EigenValues3x3(const P *array)
  {
    P a00 = array[0 * 3 + 0], a01 = array[0 * 3 + 1], a02 = array[0 * 3 + 2];
    P a10 = array[1 * 3 + 0], a11 = array[1 * 3 + 1], a12 = array[1 * 3 + 2];
    P a20 = array[2 * 3 + 0], a21 = array[2 * 3 + 1], a22 = array[2 * 3 + 2];

    P coef1 = -1;
    P coef2 = (a00 + a11 + a22);
    P coef3 = (a20 * a02) + (a10 * a01) + (a12 * a21) -
              (a00 * a11) - (a00 * a22) - (a11 * a22);

    P coef4 = (a00 * a11 * a22) + (a01 * a20 * a12) +
              (a02 * a10 * a21) - (a00 * a12 * a21) -
              (a01 * a10 * a22) - (a02 * a20 * a11);

    std::vector<P> solution = PolySolvers::SolveCubic(coef1, coef2, coef3, coef4);

    // std::sort(solution.begin(), solution.end());
    sortnet(solution);

    return solution;
  }

  // Get the eigenvector of the largest eigenvalue of a 3x3 matrix.
  // NOTE: This function will return the zero vector even if no eigenvalues exist
  template <typename P>
  std::vector<P> PrincipalEigenvector4x4(const P *M)
  {
    // First obtain the eigenvalues of M
    auto eigenvalues = EigenValues4x4(M);
    if (eigenvalues.size() == 0)
      return std::vector<P>({0, 0, 0, 0});

    P lambda = eigenvalues[eigenvalues.size() - 1]; // get the largest eigenvalue
    if (lambda == 0)
      return std::vector<P>({0, 0, 0, 0});

    // Now obtain a vector containing the M-lambda * eye(3)
    std::vector<P> A = {
      M[0] - lambda, M[1],          M[2],           M[3],
      M[4],          M[5] - lambda, M[6],           M[7],
      M[8],          M[9],          M[10] - lambda, M[11],
      M[12],         M[13],         M[14],          M[15] - lambda
    };

    // Now get the eigenvector using the Gauss-Jordan steps
    return GaussJordan4x4(A);
  }

  // Get the eigenvector of the largest eigenvalue of a 4x4 matrix.
  // NOTE: This function will return the zero vector even if no eigenvalues exist
  template <typename P>
  std::vector<P> PrincipalEigenvector3x3(const P *M)
  {
    // First obtain the eigenvalues of M
    auto eigenvalues = EigenValues3x3(M);
    if (eigenvalues.size() == 0)
      return std::vector<P>({0, 0, 0});

    P lambda = eigenvalues[eigenvalues.size() - 1]; // get the largest eigenvalue
    // if there are only zero eigenvalues, return the zero vector
    if (lambda == 0)
      return std::vector<P>({0, 0, 0}); // extra check

    // Now obtain a vector containing the M-lambda * eye(3)
    std::vector<P> A = {
      M[0] - lambda, M[1],          M[2],
      M[3],          M[4] - lambda, M[5],
      M[6],          M[7],          M[8] - lambda
    };

    // Now get the eigenvector
    return detail::computeEigenVector3(A);
  }

  // 3x3 Matrix eigen decomposition
  template <typename P>
  std::pair<std::vector<P>, std::vector<std::vector<P>>> EigenDecompose3x3(const P *M)
  {
    static_assert(std::is_floating_point<P>::value, "EigenDecompose3x3 requires a floating point type!");

    // First obtain the eigenvalues of M
    auto eigenvalues = EigenValues3x3(M);

    // if failed, return an empty eigenvalue list and empty eigenvectors
    if (eigenvalues.size() == 0)
      return std::pair<std::vector<P>, std::vector<std::vector<P>>>(eigenvalues, std::vector<std::vector<P>>{});

    std::vector<P> A = {
      M[0], M[1], M[2],
      M[3], M[4], M[5],
      M[6], M[7], M[8]
    };

    // vector of eigenvectors
    std::vector<std::vector<P>> eigenvectors;
    eigenvectors.reserve(3);
    for (size_t i = 0; i < eigenvalues.size(); i++)
    {
      const P eigval = eigenvalues[i];

      // Now subtract the eigenvalue from the diagonal
      A[0] -= eigval;
      A[4] -= eigval;
      A[8] -= eigval;

      // compute and add the eigen vector to the list
      eigenvectors.emplace_back(detail::computeEigenVector3(A));

      // add the eigenvalue back to the diagonal in order to undo the change in the elements of A
      A[0] += eigval;
      A[4] += eigval;
      A[8] += eigval;
    }

    // return the decomposition (in ascending eigenvalue order)
    return std::pair<std::vector<P>, std::vector<std::vector<P>>>(eigenvalues, eigenvectors);
  }

  // 4x4 Matrix eigen decomposition
  template <typename P>
  std::pair<std::vector<P>, std::vector<std::vector<P>>> EigenDecompose4x4(const P *M)
  {
    static_assert(std::is_floating_point<P>::value, "EigenDecompose4x4 requires a floating point type!");

    // First obtain the eigenvalues of M
    auto eigenvalues = EigenValues4x4(M);

    // if failed, return an empty eigenvalue list and empty eigenvectors
    if (eigenvalues.size() == 0)
      return std::pair<std::vector<P>, std::vector<std::vector<P>>>(eigenvalues, std::vector<std::vector<P>>{});

    std::vector<P> A = {
      M[0], M[1], M[2], M[3],
      M[4], M[5], M[6], M[7],
      M[8], M[9], M[10], M[11],
      M[12], M[13], M[14], M[15]
    };

    // vector of eigenvectors
    std::vector<std::vector<P>> eigenvectors;
    eigenvectors.reserve(4);
    for (size_t i = 0; i < eigenvalues.size(); i++)
    {
      const P eigval = eigenvalues[i];

      // Now subtract the eigenvalue from the diagonal
      A[0] -= eigval;
      A[5] -= eigval;
      A[10] -= eigval;
      A[15] -= eigval;

      // compute and add the eigen vector to the list
      eigenvectors.emplace_back(GaussJordan4x4(A));

      // add the eigenvalue back to the diagonal in order to undo the change in the elements of A
      A[0] += eigval;
      A[5] += eigval;
      A[10] += eigval;
      A[15] += eigval;
    }

    // return the decomposition (in ascending eigenvalue order)
    return std::pair<std::vector<P>, std::vector<std::vector<P>>>(eigenvalues, eigenvectors);
  }

}

#endif

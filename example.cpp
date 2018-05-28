/*
 * Examples of using the eigen34 namespace.
 * 
 * George Terzakis 2018
 * 
 */

#include <random>
#include <iostream>
#include <vector>

#include "PolySolvers.h"
#include "EigenDecompose.h"

using namespace std;


int main()
{

  // 1. Eigen-decomposition for a 3x3 matrix
  float A[] = { 1  ,  5  ,  2,
	        0  ,  1  ,  0,
	       -1  ,  7  , 11};
	       
  auto decomposition3x3 = Eigen34::EigenDecompose3x3(A);
  
  cout << " Eigen-decomposition of a 3x3 matrix" << endl << " --------------------" << endl << endl;
  cout << "A = " << endl << "  [ " << A[0] << " , " << A[1] << " , " << A[2] << " ; " << endl
                         << "    " << A[3] << " , " << A[4] << " , " << A[5] << " ; " << endl 
                         << "    " << A[6] << " , " << A[7] << " , " << A[8] << " ] " << endl << endl;
  // print the eigenvalues and respective eigenvectors
  for (int i = 0; i < decomposition3x3.first.size(); i++)
  {
    cout <<" Eigenvalue : " << decomposition3x3.first[i] <<
	   " | Eigenvector : [ " << decomposition3x3.second[i][0] <<  " , " << decomposition3x3.second[i][1] <<  " , " << decomposition3x3.second[i][2] << " ]" <<  endl; 
  }
  
  
  
  
  // 2. Eigen-decomposition for a 4x4 matrix
  float B[] = { 1  ,  5  ,  2  , -1,
		0  ,  1  ,  0  ,  8,
	       -1  ,  7  ,  11 ,  -2,
		1  , -1  ,  3  , 11 };
  auto decomposition4x4 = Eigen34::EigenDecompose4x4(B);
  cout << endl << " Eigen-decomposition of a 4x4 matrix" << endl << " --------------------" << endl;
  cout << "C = " << endl << "  [ " << B[0] << " , " << B[1] << " , " << B[2] << " , " << B[3] << " ; " << endl
                         << "    " << B[4] << " , " << B[5] << " , " << B[6] << " , " << B[7] << " ; " << endl 
                         << "    " << B[8] << " , " << B[9] << " , " << B[10] << " , " << B[11] << " ; " << endl
                         << "    " << B[12] << " , " << B[13] << " , " << B[14] << " , " << B[15] << " ] " << endl << endl;
  
  // print the eigenvalues and respective eigenvectors
  for (int i = 0; i < decomposition4x4.first.size(); i++)
  {
    cout <<" Eigenvalue : " << decomposition4x4.first[i] 
	 <<" | Eigenvector : [ " << decomposition4x4.second[i][0] <<  " , " << decomposition4x4.second[i][1] <<  " , " << decomposition4x4.second[i][2] 
	   <<" , " <<decomposition4x4.second[i][3] <<" ]" <<  endl; 
  }
  
  
  // 3. Eigen-decomposition for a 4x4 positive semi-definite matrix
  float C[] = { 0.64444  , 0.12222  , 0.53333 ,  1.25556,
	        0.12222  , 0.41111  , 0.56667 ,  0.67778,
	        0.53333  , 0.56667  , 1.00000 ,  1.56667,
	        1.25556  , 0.67778  , 1.56667 ,  2.94444 };
  cout << endl << " Eigen-decomposition of a 4x4 Positive Semidefinite Matrix" << endl << " ----------------------------------------------------" << endl;
  
  cout << "C = " << endl << "  [ " << C[0] << " , " << C[1] << " , " << C[2] << " , " << C[3] << " ; " << endl
                         << "    " << C[4] << " , " << C[5] << " , " << C[6] << " , " << C[7] << " ; " << endl 
                         << "    " << C[8] << " , " << C[9] << " , " << C[10] << " , " << C[11] << " ; " << endl
                         << "    " << C[12] << " , " << C[13] << " , " << C[14] << " , " << C[15] << " ] " << endl << endl;
                         
                         
  
  decomposition4x4 = Eigen34::EigenDecompose4x4(C);
  // print the eigenvalues and respective eigenvectors
  for (int i = 0; i < decomposition4x4.first.size(); i++)
  {
    cout <<" Eigenvalue : " << decomposition4x4.first[i] 
	 <<" | Eigenvector : [ " << decomposition4x4.second[i][0] <<  " , " << decomposition4x4.second[i][1] <<  " , " << decomposition4x4.second[i][2] 
	   <<" , " <<decomposition4x4.second[i][3] <<" ]" <<  endl; 
  }
  
  
    
  
 return 1;   
}

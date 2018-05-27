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
  
  cout << " 3x3 Decomposition" << endl << " --------------------" << endl;
  // print the eigenvalues and respective eigenvectors
  for (int i = 0; i < decomposition3x3.first.size(); i++)
  {
    cout <<" Eigenvalue : " << decomposition3x3.first[i] <<
	   " Eigenvector : [ " << decomposition3x3.second[i][0] <<  " , " << decomposition3x3.second[i][1] <<  " , " << decomposition3x3.second[i][2] << " ]" <<  endl; 
  }
  
  // 2. Eigen-decomposition for a 3x3 matrix
  float B[] = { 1  ,  5  ,  2  , -1,
		0  ,  1  ,  0  ,  8,
	       -1  ,  7  ,  11 ,  -2,
		1  , -1  ,  3  , 11 };
  auto decomposition4x4 = Eigen34::EigenDecompose4x4(B);
  cout << endl << " 4x4 Decomposition" << endl << " --------------------" << endl;
  // print the eigenvalues and respective eigenvectors
  for (int i = 0; i < decomposition4x4.first.size(); i++)
  {
    cout <<" Eigenvalue : " << decomposition4x4.first[i] 
	 <<" Eigenvector : [ " << decomposition4x4.second[i][0] <<  " , " << decomposition4x4.second[i][1] <<  " , " << decomposition4x4.second[i][2] 
	   <<" , " <<decomposition4x4.second[i][3] <<" ]" <<  endl; 
  }
  
  
  // 2. Eigen-decomposition for a 3x3 matrix
  float C[] = { 0.64444  , 0.12222  , 0.53333 ,  1.25556,
	        0.12222  , 0.41111  , 0.56667 ,  0.67778,
	        0.53333  , 0.56667  , 1.00000 ,  1.56667,
	        1.25556  , 0.67778  , 1.56667 ,  2.94444 };
  decomposition4x4 = Eigen34::EigenDecompose4x4(C);
  cout << endl << " 4x4 Decomposition" << endl << " --------------------" << endl;
  // print the eigenvalues and respective eigenvectors
  for (int i = 0; i < decomposition4x4.first.size(); i++)
  {
    cout <<" Eigenvalue : " << decomposition4x4.first[i] 
	 <<" Eigenvector : [ " << decomposition4x4.second[i][0] <<  " , " << decomposition4x4.second[i][1] <<  " , " << decomposition4x4.second[i][2] 
	   <<" , " <<decomposition4x4.second[i][3] <<" ]" <<  endl; 
  }
 return 1;   
}

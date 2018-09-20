//------------------------------------------------------------------------------
/// \file Matrix_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Main driver function for Matrix class with copy ctor/copy
/// assignment.
/// \ref    : 17.5.1 Copy Ch. 17 Construction, Cleanup, Copy, and Move; 
///   Bjarne Stroustrup, The C++ Programming Language, 4th Ed., Stroustrup    
/// \details RAII for C-style arrays. order-2 matrices (rows and columns)
/// \copyright If you find this code useful, feel free to donate directly and
/// easily at this direct PayPal link: 
///
/// https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
/// 
/// which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.
/// Otherwise, I receive emails and messages on how all my (free) material on
/// physics, math, and engineering have helped students with their studies, and
/// I know what it's like to not have money as a student, but love physics (or 
/// math, sciences, etc.), so I am committed to keeping all my material 
/// open-source and free, whether or not sufficiently crowdfunded, under the 
/// open-source MIT license: feel free to copy, edit, paste, make your own
/// versions, share, use as you wish.    
/// Peace out, never give up! -EY
//------------------------------------------------------------------------------
/// COMPILATION TIPS:
///  g++ -std=c++14 Matrix_main.cpp -o Matrix_main
//------------------------------------------------------------------------------
#include "Matrix.h"

#include <iostream>
#include <utility> // std::move

using Modules::Matrices::Matrix;

int main()
{
  // MatrixConstructsHadamardMatrices
  Matrix<int> H2 {2, 2, {1, 1, 1, -1}};
  std::cout << H2 << '\n';

  Matrix<int> H4 {4, 4, {1, 1, 1, 1, 
                        1, -1, 1, -1,
                        1, 1, -1, -1, 
                        1, -1, -1, 1}};

  std::cout << H4 << '\n';                  

  // MatrixCopiesHadamardMatrices
  Matrix<int> H2_copy1 {H2};
  std::cout << H2_copy1 << '\n';

  Matrix<int> H4_copy1 = H4;
  std::cout << H4_copy1 << '\n';

  // Check that copies are "independent".
  H2.set_element(1, 1, 2);
  std::cout << H2 << '\n';
  H2_copy1.set_element(0, 1, 3);
  std::cout << H2_copy1 << '\n';

  // MatrixMoveConstructs.
  Matrix<int> H2_2 {std::move(H2)};
  std::cout << H2_2 << '\n';
  std::cout << H2.dim()[0] << H2.dim()[1] << '\n';

  // MatrixMoveAssigns.
  Matrix<int> H2_3 = {std::move(H2_2)};
  std::cout << H2_3 << '\n';
  std::cout << H2_2.dim()[0] << H2_2.dim()[1] << '\n';

  // MatricesAdd.
  Matrix<int> G4 {4, 4, {0, 1, 2, 3, 
                        4, 5, 6, 7,
                        -1, -2, -3, -4, 
                        -5, -6, -7, -8}};

  Matrix<int> I4 {G4 + H4};
  std::cout << I4 << '\n';

  I4 += H4;
  std::cout << I4 << '\n';

  
}

/**
 * @file   : stdcopy_eg.cpp
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : Examples of std::copy.  
 * @details Copies elements in the range. 
 *   1st, I'll try copying values from a pointer array of size 5 (or N) to another 
 * 	 pointer array of size 6 (or N + 1)
 * @ref    : http://en.cppreference.com/w/cpp/algorithm/copy
 * If you find this code useful, feel free to donate directly and easily at this direct PayPal link: 
 * 
 * https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
 * 
 * which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.  
 * Otherwise, I receive emails and messages on how all my (free) material on physics, math, and engineering have 
 * helped students with their studies, and I know what it's like to not have money as a student, but love physics 
 * (or math, sciences, etc.), so I am committed to keeping all my material open-source and free, whether or not 
 * sufficiently crowdfunded, under the open-source MIT license: 
 *  feel free to copy, edit, paste, make your own versions, share, use as you wish.    
 * Peace out, never give up! -EY
 * 
 * COMPILATION TIPS:
 *  g++ -std=c++14 FileOpen.cpp FileOpen_main.cpp -o FileOpen_main
 * */
#include <algorithm> 	// std::copy
#include <iostream>

#include <memory> 		// std::unique_ptr, std::make_unique

int main()
{


  double* ptr_to_5_elements {new double[5]};
  for (int i {1}; i < 6; i++)
  {
    ptr_to_5_elements[i-1] = i;
  }
  for (int i {0}; i < 5; i++)
  {
    std::cout << ptr_to_5_elements[i] << " ";
  } std::cout << '\n' << '\n';

  double* ptr_to_6_elements {new double[6]};

  std::copy(
  	&ptr_to_5_elements[0],
  	&ptr_to_5_elements[5],
  	&ptr_to_6_elements[0]);

/*  for (int i {0}; i < 5; i++)
  {
    ptr_to_6_elements[i] = ptr_to_5_elements[i];
    std::cout << ptr_to_6_elements[i] << " ";
  } std::cout << std::endl;
*/

  ptr_to_6_elements[5] = 7.;
  for (int i {0}; i < 6; i++)
  {
    std::cout << ptr_to_6_elements[i] << " ";
  } std::cout << '\n' << '\n';

//  std::unique_ptr<double[]> uptr_to_6_elements {std::make_unique<double[]>(6)};

  //---------------------------------------------------------------------------
  /// \details We can even move-initialize a unique_ptr with a raw pointer!
  //--------------------------------------------------------------------------- 
  std::unique_ptr<double[]> uptr_to_6_elements (ptr_to_6_elements);
  for (int i {0}; i < 6; i++)
  {
    std::cout << uptr_to_6_elements[i] << " ";
  } std::cout << '\n' << '\n';
	  
  std::unique_ptr<double[]> uptr_to_7_elements {std::make_unique<double[]>(7)};

  std::copy(
  	&uptr_to_6_elements[0],
  	&uptr_to_6_elements[6],
  	&uptr_to_7_elements[0]);

  std::cout << " Is uptr_to_6_elements after a copy a nullptr? " << 
  	(uptr_to_6_elements == nullptr) << '\n';

 for (int i {0}; i < 6; i++)
  {
    std::cout << uptr_to_6_elements[i] << " ";
  } std::cout << '\n' << '\n';

 for (int i {0}; i < 7; i++)
  {
    std::cout << uptr_to_7_elements[i] << " ";
  } std::cout << '\n' << '\n';

 uptr_to_7_elements[6] = 8.;

 for (int i {0}; i < 7; i++)
  {
    std::cout << uptr_to_7_elements[i] << " ";
  } std::cout << '\n' << '\n';


  delete[] ptr_to_5_elements;
//  delete[] ptr_to_6_elements;
}

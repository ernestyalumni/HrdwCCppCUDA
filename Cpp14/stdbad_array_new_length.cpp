/**
 * @file   : stdbad_array_new_length_eg.cpp
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : Example of std::bad_array_new_length.  
 * @details Object of type std::initializer_list<T>, lightweight proxy object 
 * 	that provides access to an array of objects of type const T
 * @ref    : http://en.cppreference.com/w/cpp/utility/initializer_list
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
#include <iostream>
#include <new> // std::bad_array_new_length
#include <climits>

#include <stdexcept>

int main()
{
	int negative {-1};
  int small {1};
  int large {INT_MAX};
  try 
  {
    new int[negative];    // negative size
    new int[small]{1,2,3};  // too many initializers
    new int[large][1000000];  // too large
  }
  catch(const std::bad_array_new_length &e)
  {
    std::cout << e.what() << '\n';
  }

  std::cout << "Try throw now" << '\n'; 

  if (negative < 0)
  {
//    throw std::bad_array_new_length;
 //   throw "Bad array new length";
  }

//  size_t a {-2};
}

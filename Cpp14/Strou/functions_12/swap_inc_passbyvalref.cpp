/**
 * @file   : swap_inc_passbyvalref.cpp
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : Swap to demonstrate argument passing.
 * @ref    : 12.2 Argument Passing Ch. 12 Functions; Bjarne Stroustrup, The C++ Programming Language, 4th Ed., Stroustrup  
 * pp. 47 Ch. 3 GCD; Edward Scheinerman, C++ for Mathematicians: An Introduction for Students and Professionals. Taylor & Francis Group, 2006.   
 * If you find this code useful, feel free to donate directly and easily at this direct PayPal link: 
 * 
 * https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
 * 
 * which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.  
 * Otherwise, I receive emails and messages on how all my (free) material on physics, math, and engineering have 
 * helped students with their studies, and I know what it's like to not have money as a student, but love physics 
 * (or math, sciences, etc.), so I am committed to keeping all my material open-source and free, whether or not 
 * sufficiently crowdfunded, under the open-source MIT license: 
 * 	feel free to copy, edit, paste, make your own versions, share, use as you wish.    
 * Peace out, never give up! -EY
 * 
 * */
#include <iostream>

void swap(long a, long b)
{
  long tmp;
  tmp = a;
  a = b;
  b = tmp;
}

void swap_by_ref(long& a, long& b)
{
  long tmp;
  tmp = a;
  a = b;
  b = tmp;
}

void f(int val, int& ref)
{
  ++val;
  ++ref;
}

int main(int argc, char* argv[])
{
  long a = 5;
  long b = 17; 
  swap(a, b);
  
  std::cout << " a = " << a << std::endl; // 5 
  std::cout << " b = " << b << std::endl; // 17 
  std::cout << " Didn't get swapped by pass by value. " << '\n';

  swap_by_ref(a, b);
  std::cout << " a = " << a << std::endl; // 5 
  std::cout << " b = " << b << std::endl; // 17 
  std::cout << " Swapped by pass by refernece. " << '\n';
  


}

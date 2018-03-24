/**
 * @file   : static_in_functions.cpp
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : static in functions.
 * @ref    : 12.1.8 Local Variables Ch. 12 Functions; Bjarne Stroustrup, The C++ Programming Language, 4th Ed., Stroustrup  
 * 
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

void f(int a)
{
  while(a--)
  {
      static int n = 0;     // initialized once
      int x = 0;            // initialized 'a' times in each call of f()

      std::cout << " n == " << n++ << " , x == " << x++ << '\n';
  }
}

int main()
{
  f(3);
  f(4);
}
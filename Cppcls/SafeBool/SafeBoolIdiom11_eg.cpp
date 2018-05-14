/**
 * @file   : SafeBoolIdiom11_eg.cpp
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : Demonstrate Safe bool idiom with C++11.
 * @ref    :
 * @detail : Provide boolean tests for a class but restricting it from taking 
 *  participation in unwanted expressions.
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
 *  feel free to copy, edit, paste, make your own versions, share, use as you wish.    
 * Peace out, never give up! -EY
 * 
 * COMPILATION TIPS:
 *  g++ -std=c++14 noexcept_eg.cpp -o noexcept_eg
 * */
#include <iostream>

//------------------------------------------------------------------------------
/// \details User provided boolean conversion functions can cause more harm 
///   than benefit because it allows them to participate in expressions you 
///   wouldn't ideally want them to. If a safe conversion operator is defined
///   then 2 or more objects of unrelated classes can be compared. 
///   Type safety is compromised.  
///
///   C++11 provides explicit conversion operators as a parallel to explicit 
///   constructors. This new feature solves the problem in a clean way.
//------------------------------------------------------------------------------
struct Testable
{
  explicit operator bool() const
  {
    return false;
  }
};

int main()
{
  Testable a, b;
  if (a)
  {
    std::cout << "\n a as predicate was true. \n";
  }
  else
  {
    std::cout << "\n a as predicate was false. \n";
  }

/*  if (a == b) // compiler error
  {
    /* do something */
  //}

}

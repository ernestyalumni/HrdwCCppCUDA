//------------------------------------------------------------------------------
/// \file String_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Main driver function for String class.
/// \ref Sec. 19.3 A String Class.
///   The C++ Programming Language, 4th Ed., Stroustrup; Ch. 19 Special Ops
/// \details String provides value semantics, checked and unchecked access to
/// characters, stream I/O, support for range-for loops, equality operations,
/// and concatenation operators.
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
///  g++ -std=c++14 String_main.cpp -o String_main
//------------------------------------------------------------------------------
#include "String.h"

#include <cstring> // std::strcpy 
#include <iostream>

using Utilities::String;

int main()
{
  // std::strcpyWorks
  {
    

  }

  // \ref 19.3.6 Using Our String
  {
    String s {"abcdefghij"};
    std::cout << s << '\n';
    s += 'k';
    s += 'l';
    s += 'm';
    s += 'n';
    std::cout << s << '\n';
    String s2 {"Hell"};
//    s2 += " and high water";
//    std::cout << s2 << '\n';

  }

}

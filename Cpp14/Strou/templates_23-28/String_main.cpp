//------------------------------------------------------------------------------
/// \file String_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Driver main file for class template String, a more general type
///   string that relies on fact that a character can be copied.
/// \details The basic mechanisms for defining and using class templates are
///   introduced through the example of a string template.
/// \ref 23.2 A Simple String Template Ch. 23 Templates; Bjarne Stroustrup, 
///   The C++ Programming Language, 4th Ed., Stroustrup; Ch.23  
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
///  g++ -std=c++14 FileOpen_main.cpp FileOpen.cpp -o FileOpen_main
//------------------------------------------------------------------------------
#include "String/String.h"

#include <iostream>
#include <map>
#include <string>

using Templates::Strings::String;
using Templates::Strings::BasicString;

int main() // count the occurrences of each word on input
{
//  std::map<String<char>, int> m;

//  for (String <char> buf; std::cin >> buf;)
  //{
    //++m[buf];
//  }
  // ... write out result ...

  // WORKS but doesn't terminate
//  std::map<std::string, int> m;
  //for (std::string buf; std::cin >> buf;)
//  {
  //  ++m[buf];
 // }
  // ... write out result ...
//  std::cout << 

}

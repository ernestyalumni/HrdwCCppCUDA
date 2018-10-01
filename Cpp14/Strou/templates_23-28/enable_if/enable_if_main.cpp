//------------------------------------------------------------------------------
/// \file enable_if_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Main driver function for enable_if.h.
/// \ref https://en.cppreference.com/w/cpp/types/enable_if
/// \details Demonstrate enable_if for template metaprogramming, and its
/// relationship to interfaces. 
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
///  g++ -std=c++14 enable_if_main.cpp -o enable_if_main
//------------------------------------------------------------------------------
#include "enable_if.h"

#include <string>
#include <type_traits> // std::aligned_union_t

using Templates::construct;
using Templates::destroy;

int main()
{
  // \ref https://en.cppreference.com/w/cpp/types/enable_if
  {
    std::aligned_union_t<0, int, std::string> u;

    construct(reinterpret_cast<int*>(&u));
    destroy(reinterpret_cast<int*>(&u));

    // Segmentation Fault
    construct(reinterpret_cast<std::string*>(&u), "Hello");
    destroy(reinterpret_cast<std::string*>(&u));

//    A<int> a1; // OK, matches the primary template
  //  A<double> a2; // OK, matches the partial specialization
  }
}

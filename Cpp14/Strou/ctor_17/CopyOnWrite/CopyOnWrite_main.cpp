//------------------------------------------------------------------------------
/// \file CopyOnWrite_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Main driver function for CopyOnWrite class with copy-on-write idiom
/// \ref    : 17.5.1.3 Copy Ch. 17 Construction, Cleanup, Copy, and Move; 
///   Bjarne Stroustrup, The C++ Programming Language, 4th Ed., Stroustrup    
/// https://en.wikibooks.org/wiki/More_C%2B%2B_Idioms/Copy-on-write
/// \details Drawback; classes that return references to their internal state.
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

#include "CopyOnWrite.h"

#include <iostream>
#include <string>

using Idioms::CopyOnWrite;

int main()
{
  CopyOnWrite<int> copy_on_write_int_5 {new int[5]};

  CopyOnWrite<char*> s1 = "Hello";
  std::cout << *s1 << '\n';

  char &c = s1->operator[](4); // Non-const detachment does nothing here
  CopyOnWrite<char> s2(s1); // Lazy-copy, shared state.

}

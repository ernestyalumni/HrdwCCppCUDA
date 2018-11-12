//------------------------------------------------------------------------------
/// \file CheckReturn_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Main driver file for helper functions that check system calls.
/// \ref      
/// \details C++ Functors
/// \copyright If you find this code useful, feel free to donate directly and easily at 
/// this direct PayPal link: 
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
///  g++ -std=c++14 CheckReturn_main.cpp -o CheckReturn_main
//------------------------------------------------------------------------------
#include "CheckReturn.h"

#include <iostream>

using Utilities::CheckClose;
using Utilities::CheckReturn;

int main()
{
  // CheckReturnFunctionCallReturnsNonNegativeValues
  {
    std::cout << " \n CheckReturnFunctionCallReturnsNonNegativeValues \n";

    std::cout << CheckReturn()(5, "input argument e was not 0 or positive") <<
      '\n';
  }

  // CheckCloseFunctionCallReturnsNonNegativeValues
  {
    std::cout << " \n CheckCloseFunctionCallReturnsNonNegativeValues \n";

    std::cout << CheckClose()(4) << '\n';
  }
}

//------------------------------------------------------------------------------
/// \file EventFd_main.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Main driver file for eventfd, for event notification.
/// \ref http://man7.org/linux/man-pages/man2/eventfd.2.html  
/// \details 
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
///  g++ -std=c++14 EventFd_main.cpp -o EventFd_main
//------------------------------------------------------------------------------
#include "EventFd.h"

#include <iostream>

using IO::EventFdFlags;

int main()
{
  // EventFdFlagsIsAnEnumClass
  {
    std::cout << "\n EventFdFlagsIsAnEnumClass \n";  
    std::cout << " EventFdFlags::default_value : " << 
      static_cast<int>(EventFdFlags::default_value) << '\n'; // 0
    std::cout << " EventFdFlags::close_on_execute : " << 
      static_cast<int>(EventFdFlags::close_on_execute) << '\n'; // 524288
    std::cout << " EventFdFlags::non_blocking : " << 
      static_cast<int>(EventFdFlags::non_blocking) << '\n'; // 2048
    std::cout << " EventFdFlags::semaphore : " << 
      static_cast<int>(EventFdFlags::semaphore) << '\n'; // 1
  }



}

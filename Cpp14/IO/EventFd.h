//------------------------------------------------------------------------------
/// \file EventFd.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  A file descriptor (fd) for event notification.
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
#ifndef _IO_EVENT_FD_H_
#define _IO_EVENT_FD_H_

#include "../Utilities/ensure_valid_results.h" // check_valid_fd

#include <iostream>
#include <sys/eventfd.h>
#include <unistd.h> // ::read, ::close

namespace IO
{

//------------------------------------------------------------------------------
/// \brief enum class for all eventfd flags, that maybe bitwise ORed in to
/// change behavior of eventfd().
//------------------------------------------------------------------------------
enum class EventFdFlags : int
{
  default_value = 0,
  close_on_execute = EFD_CLOEXEC,
  non_blocking = EFD_NONBLOCK,
  semaphore = EFD_SEMAPHORE
};



} // namespace IO

#endif // _IO_EVENT_FD_H_
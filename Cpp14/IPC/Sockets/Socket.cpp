//------------------------------------------------------------------------------
/// \file Socket.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  A socket as RAII, source file 
/// \ref      
/// \details Using RAII for socket. 
/// \copyright If you find this code useful, feel free to donate directly
/// (username ernestyalumni or email address above), going directly to: 
///
/// paypal.me/ernestyalumni
/// 
/// which won't go through a 3rd. party like indiegogo, kickstarter, patreon.
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
///   g++ -I ../../ -std=c++14 Socket.cpp Socket_main.cpp ../../Utilities/ErrorHandling.cpp -o Socket_main
//------------------------------------------------------------------------------
#include "Socket.h"

#include <unistd.h> // ::close

#include "Utilities/ErrorHandling.h" // HandleReturnValue

using Utilities::ErrorHandling::HandleClose;
using Utilities::ErrorHandling::HandleReturnValue;

namespace IPC
{
namespace Sockets
{

Socket::Socket(const Domains domain, const int type, const int protocol) :
  domain_{domain},
  type_{type},
  protocol_{protocol},
  fd_{::socket(static_cast<uint16_t>(domain), type, protocol)}
{
  HandleReturnValue()(fd_, "create file descriptor (::eventfd)");
}

Socket::~Socket()
{
  const int result {::close(fd_)};
  HandleClose()(result);
}



} // namespace Sockets
} // namespace IPC
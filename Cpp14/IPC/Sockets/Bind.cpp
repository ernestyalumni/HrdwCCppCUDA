//------------------------------------------------------------------------------
/// \file Bind.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Wrapper for ::bind as a C++ functor.
/// \ref http://man7.org/linux/man-pages/man2/bind.2.html
/// \details Event describes object linked to fd.  struct epoll_event.
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
///   g++ --std=c++17 -I ../../ Event.cpp Event_main.cpp -o Event_main
//------------------------------------------------------------------------------
#include "InternetAddress.h"
#include "Socket.h"
#include "Utilities/ErrorHandling.h" // HandleReturnValue

namespace IPC
{

namespace Sockets
{

} // namespace Sockets
} // namespace IPC
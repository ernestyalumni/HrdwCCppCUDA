//------------------------------------------------------------------------------
/// \file Listen.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Wrapper for ::listen as a C++ functor.
/// \ref http://man7.org/linux/man-pages/man2/bind.2.html
/// \details Listen for connections on a socket.
/// To accept connections,
/// 1. Socket created with ::socket
/// 2. Socket bound to local address using ::bind, so other sockets may be
/// ::connect 'ed to it
/// 3. Willingness to accept incoming connections and a queue limit for incoming
/// connections are specified with ::listen()
/// 4. Connections accepted with ::accept().
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
///   g++ -I ../../ -std=c++14 Socket.cpp Listen_main.cpp \
///     ../../Utilities/ErrorHandling.cpp ../../Utilities/Errno.cpp -o \
///     Listen_main
//------------------------------------------------------------------------------
#include "Listen.h"

#include "Socket.h"
#include "Utilities/ErrorHandling.h" // HandleReturnValue

namespace IPC
{

namespace Sockets
{

Listen::Listen() = default;

Listen::Listen(const int backlog):
  backlog_{backlog}
{}

void Listen::operator()(Socket& socket)
{
  const int sockfd {socket.fd()};

  const int result {::listen(sockfd, backlog_)};

  HandleListen()(result);
}

void Listen::operator()(Socket& socket, const int backlog)
{
  backlog_ = backlog;

  this->operator()(socket);
}

} // namespace Sockets
} // namespace IPC

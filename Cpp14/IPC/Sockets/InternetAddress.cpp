//------------------------------------------------------------------------------
/// \file InternetAddress.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Wrappers and classes for socket addresses.
/// \ref http://man7.org/linux/man-pages/man7/ip.7.html
/// \details Only 1 IP (Internet Protocol) socket may be bound to any given
/// local (address, port) pair.
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

#include <arpa/inet.h> // htonl, htons
#include <netinet/ip.h> // ::sockaddr_in
#include <ostream>

namespace IPC
{

namespace Sockets
{

InternetSocketAddress::InternetSocketAddress(
  const uint16_t sin_port,
  const uint16_t sin_family,
  const uint32_t sin_addr
  ):
  ::sockaddr_in{sin_family, ::htons(sin_port), {::htonl(sin_addr)}}
{}

InternetSocketAddress::InternetSocketAddress() = default;

InternetAddress::InternetAddress(
  const uint16_t sin_family,
  const uint16_t sin_port,
  const uint32_t sin_addr
  ):
  socket_address_internet_{sin_family, ::htons(sin_port), ::htonl(sin_addr)}
{}

InternetAddress::InternetAddress(const uint16_t sin_port):
  InternetAddress{AF_INET, sin_port, INADDR_ANY}
{}

InternetAddress::InternetAddress() = default;

} // namespace Sockets
} // namespace IPC
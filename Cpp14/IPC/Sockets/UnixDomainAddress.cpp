//------------------------------------------------------------------------------
/// \file UnixDomainAddress.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Wrappers and classes for UNIX domain socket addresses.
/// \ref http://man7.org/linux/man-pages/man7/unix.7.html
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
#include "UnixDomainAddress.h"

#include "Socket.h"
#include "Utilities/casts.h" // get_underlying

#include <cstring> // std::memset, std::strncpy, std::strlen
#include <netinet/un.h>
#include <ostream>
#include <unistd.h> // unlink

using IPC::Sockets::Domains;
using Utilities::get_underlying_value;

namespace IPC
{

namespace Sockets
{

UnixDomainSocketAddress::UnixDomainSocketAddress(
  const uint16_t sun_family,
  const std::string& sun_path
  ):
{
  this->sun_family = sun_family;
  std::strcpy(this->sun_path, sun_path.c_str());

  /// \url https://github.com/skuhl/sys-prog-examples/blob/master/ \
  /// simple-examples/unix-dgram-server.c
  /// Notice that we called ::unlink() before ::bind() to remove the socket if
  /// it already exists. You'll get an EINVAL error if file is already there.
  ::unlink(this->sun_path);
}

UnixDomainSocketAddress::UnixDomainSocketAddress(const std::string& sun_path):
  UnixDomainSocketAddress{
    get_underlying_value<Domains>(Domains::unix),
    sun_path}
{}

UnixDomainSocketAddress::UnixDomainSocketAddress() = default;

UnixDomainAddress::UnixDomainAddress(
  const uint16_t sun_family,
  const std::string& sun_path
  ):
{
  socket_address_unix_.sun_family = sun_family;
  std::strcpy(socket_address_unix_.sun_path, sun_path.c_str());

  ::unlink(socket_address_unix_.sun_path);
}

UnixDomainAddress::UnixDomainAddress(const std::string& sun_path):
  UnixDomainAddress{get_underlying_value<Domains>(Domains::unix), sun_path}
{}

UnixDomainAddress::UnixDomainAddress() = default;

} // namespace Sockets
} // namespace IPC
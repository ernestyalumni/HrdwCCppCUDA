//------------------------------------------------------------------------------
/// \file InternetAddress.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Wrappers and classes for socket addresses.
/// \ref http://man7.org/linux/man-pages/man7/ip.7.html
/// \details Only 1 IP (Internet Protocol) socket may be bound to any given
/// local (address, port) pair.
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

// htonl converts a long interger (e.g. address) to a network representation
// (IP-standard byte ordering).
InternetSocketAddress::InternetSocketAddress(
  const uint16_t sin_port,
  const uint16_t sin_family,
  const uint32_t sin_addr
  ):
  ::sockaddr_in{sin_family, ::htons(sin_port), {::htonl(sin_addr)}}
{}

InternetAddress::InternetAddress(
  const uint16_t sin_port,
  const uint16_t sin_family,
  const uint32_t sin_addr
  ):
  socket_address_internet_{sin_family, ::htons(sin_port), ::htonl(sin_addr)}
{}

} // namespace Sockets
} // namespace IPC
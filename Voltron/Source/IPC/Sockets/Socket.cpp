//------------------------------------------------------------------------------
/// \file Socket.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref https://linux.die.net/man/2/socket
//------------------------------------------------------------------------------
#include "Socket.h"

#include "Cpp/Utilities/TypeSupport/UnderlyingTypes.h"
#include "Utilities/ErrorHandling/ErrorHandling.h"

#include <unistd.h> // ::close

using Cpp::Utilities::TypeSupport::get_underlying_value;
//using Utilities::ErrorHandling::HandleClose;

namespace IPC
{
namespace Sockets
{

Socket::Socket(const int domain, const int type, const int protocol):
  domain_{domain},
  type_{type},
  protocol_{protocol},
  fd_{create_socket(domain_, type_, protocol_)}
{}

Socket::Socket(const Domain domain, const Type type, const int protocol):
  Socket{get_underlying_value(domain), get_underlying_value(type), protocol}
{}

Socket::~Socket()
{
  const int result {::close(fd_)};
  //HandleClose()(result);
}

int Socket::create_socket(const int domain, const int type, const int protocol)
{
  const int result {::socket(domain, type, protocol)};

  //HandleSocket()(result);

  return result;
}

/*
Socket::HandleSocket::HandleSocket() = default;

void Socket::HandleSocket::operator()(const int result)
{ 
  this->operator()(result, "create socket file descriptor (::socket)");
}
*/

std::ostream& operator<<(std::ostream& os, const Socket& socket)
{
  os << socket.domain() << ' ' << socket.type() << ' ' <<
    socket.protocol() << ' ' << socket.fd() << '\n';

  return os;
}

} // namespace Sockets
} // namespace IPC

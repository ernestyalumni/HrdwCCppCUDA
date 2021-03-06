//------------------------------------------------------------------------------
/// \file CreateSocket.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref https://www.cs.rutgers.edu/~pxk/417/notes/sockets/demo-udp-03.html
//------------------------------------------------------------------------------
#include "CreateSocket.h"

#include "IPC/Sockets/Bind.h"
#include "IPC/Sockets/ParameterFamilies.h"
#include "IPC/Sockets/Receive.h"
#include "IPC/Sockets/Socket.h"

#include <optional>
#include <utility> // std::move

using IPC::Sockets::Domain;
using IPC::Sockets::ReceivingOn;
using IPC::Sockets::Socket;
using IPC::Sockets::Type;
using Utilities::ErrorHandling::ErrorNumber;

namespace IPC
{
namespace Sockets
{
namespace UDP
{

UdpSocket::UdpSocket(const Domain domain):
  Socket{domain, Type::datagram}
{}

std::pair<
  std::optional<ErrorNumber>,
  std::optional<InternetSocketAddress>
  > bind_to_any_ip_address(Socket& socket, const uint16_t port)
{
  const InternetSocketAddress internet_socket_address {port};

  Bind bind_f {internet_socket_address};

  std::optional<ErrorNumber> bind_result {bind_f(socket)};
  
  if (bind_result) 
  {
    return std::make_pair<
      std::optional<ErrorNumber>,
      std::optional<InternetSocketAddress>
      >(std::move(bind_result), std::nullopt);
  }
  else
  {
    return std::make_pair<
      std::optional<ErrorNumber>,
      std::optional<InternetSocketAddress>
      >(
        std::nullopt,
        std::make_optional<InternetSocketAddress>(
          std::move(internet_socket_address)));
  }
}

} // namespace UDP
} // namespace Sockets
} // namespace IPC

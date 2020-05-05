//------------------------------------------------------------------------------
/// \file Receiver.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref https://www.cs.rutgers.edu/~pxk/417/notes/sockets/demo-udp-03.html
//------------------------------------------------------------------------------
#include "Receiver.h"

#include "IPC/Sockets/UDP/CreateSocket.h"

namespace IPC
{
namespace Sockets
{
namespace UDP
{

Receiver::Receiver(const uint16_t port):
  socket_{std::move(UdpSocket{})}
{
  auto bind_result = bind_to_any_ip_address(socket_, port);
  // TODO: Handle bind failure.
}

} // namespace UDP
} // namespace Sockets
} // namespace IPC

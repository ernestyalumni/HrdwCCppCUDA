//------------------------------------------------------------------------------
/// \file CreateSocket.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref https://www.cs.rutgers.edu/~pxk/417/notes/sockets/demo-udp-03.html
//------------------------------------------------------------------------------
#ifndef IPC_SOCKETS_UDP_CREATE_SOCKET_H
#define IPC_SOCKETS_UDP_CREATE_SOCKET_H

#include "IPC/Sockets/InternetAddress.h"
#include "IPC/Sockets/ParameterFamilies.h"
#include "IPC/Sockets/Socket.h"
#include "Utilities/ErrorHandling/ErrorNumber.h"

#include <optional>
#include <utility> // std::pair

using IPC::Sockets::Domains;

namespace IPC
{
namespace Sockets
{
namespace UDP
{

// Create a UDP socket.

class UdpSocket : public Socket
{
	public:
		
		explicit UdpSocket(const Domains domain = Domains::ipv4);
};

//------------------------------------------------------------------------------
/// \fn bind_to_any_ip_address
/// \brief Create any local address and bind an incoming, input socket to the
/// address; return the address if successful.
//------------------------------------------------------------------------------
std::pair<
	std::optional<Utilities::ErrorHandling::ErrorNumber>,
	std::optional<InternetSocketAddress>
	> bind_to_any_ip_address(Socket& socket, const uint16_t port);

} // namespace UDP
} // namespace Sockets
} // namespace IPC

#endif // IPC_SOCKETS_UDP_CREATE_SOCKET_H

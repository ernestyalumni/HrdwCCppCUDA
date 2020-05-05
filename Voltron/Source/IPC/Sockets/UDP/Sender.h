//------------------------------------------------------------------------------
/// \file Sender.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref https://www.cs.rutgers.edu/~pxk/417/notes/sockets/demo-udp-03.html
//------------------------------------------------------------------------------
#ifndef IPC_SOCKETS_UDP_SENDER_H
#define IPC_SOCKETS_UDP_SENDER_H

#include "IPC/Sockets/InternetAddress.h"
#include "IPC/Sockets/Send.h"
#include "IPC/Sockets/Socket.h"
#include "Utilities/ErrorHandling/ErrorNumber.h"

#include <iostream>

#include <arpa/inet.h>

#include <array>
#include <optional>
#include <string>
#include <utility> // std::pair

namespace IPC
{
namespace Sockets
{
namespace UDP
{

class Sender
{
  public:

		explicit Sender(const uint16_t port = 0);

		template <std::size_t N = 2048>
		auto operator()(const std::string& destination_address, const int flags = 0)
		{
			SendTo<N> send_to {flags};

			InternetSocketAddress address;

			//std::optional<InternetSocketAddress> recipient_address {
				//address_to_network_binary(destination_address, address)};

			::inet_aton(destination_address.c_str(), &address.sin_addr);

			/*
			if (!(recipient_address))
			{
				std::cout << "inet_aton() failed";
			}
			*/
			InternetSocketAddress recipient_address {address};

//			typename SendTo<N>::SendMessage send_message {send_to(*recipient_address)};
			typename SendTo<N>::SendMessage send_message {send_to(recipient_address)};

			Socket& socket {socket_};

			auto send_on_socket =
				[&socket, &send_message](const std::array<char, N>& buffer)
				{
					return send_message(buffer)(socket);
				};

			return send_on_socket;
		}

  private:

		Socket socket_;
};

} // namespace UDP
} // namespace Sockets
} // namespace IPC

#endif // IPC_SOCKETS_UDP_SENDER_H

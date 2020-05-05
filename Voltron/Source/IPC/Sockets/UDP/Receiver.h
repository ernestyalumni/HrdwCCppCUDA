//------------------------------------------------------------------------------
/// \file Receiver.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref https://www.cs.rutgers.edu/~pxk/417/notes/sockets/demo-udp-03.html
//------------------------------------------------------------------------------
#ifndef IPC_SOCKETS_UDP_RECEIVER_H
#define IPC_SOCKETS_UDP_RECEIVER_H

#include "IPC/Sockets/Receive.h"
#include "IPC/Sockets/Socket.h"
#include "Utilities/ErrorHandling/ErrorNumber.h"

#include <optional>
#include <utility> // std::pair

namespace IPC
{
namespace Sockets
{
namespace UDP
{

template <std::size_t N = 2048>
auto receive_data(Socket& socket)
{
  IPC::Sockets::ReceivingOn receiving_on {};

  return receiving_on.operator()<N>(socket)();
}

class Receiver
{
	public:

		using ErrorNumber = Utilities::ErrorHandling::ErrorNumber;

		template <std::size_t N>
		using Receipt = typename IPC::Sockets::ReceivingOn::ReceivedFrom<N>::Receipt;

		Receiver(const uint16_t port);

		template <std::size_t N = 2048>
		std::pair<std::optional<ErrorNumber>, std::optional<Receipt<N>>> receiving_data()
		{
			return receive_data<N>(socket_);
		}

	private:

		Socket socket_;
};

} // namespace UDP
} // namespace Sockets
} // namespace IPC

#endif // IPC_SOCKETS_UDP_RECEIVER_H

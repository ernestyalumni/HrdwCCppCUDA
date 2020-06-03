//------------------------------------------------------------------------------
/// \file MainUDP-Receiver.cpp
/// \author Ernest Yeung
/// \brief Main file to be a UDP receiver.
///
/// \details int main() is needed.
/// \ref
///-----------------------------------------------------------------------------
#include "IPC/Sockets/UDP/Receiver.h"

#include "IPC/Sockets/Receive.h"
#include "IPC/Sockets/Send.h"
#include "IPC/Sockets/UDP/CreateSocket.h"

#include "Utilities/ToBytes.h"

#include <array>
#include <cstdlib> // std::atoi
#include <iostream>
#include <string>
#include <utility>

using IPC::Sockets::UDP::Receiver;

using IPC::Sockets::ReceivingOn;
using IPC::Sockets::SendTo;
using IPC::Sockets::Socket;
using IPC::Sockets::UDP::bind_to_any_ip_address;
using IPC::Sockets::UDP::UdpSocket;
using Utilities::ToBytes;

int main(int argc, char* argv[])
{
	// Arbitrarily chosen, hard-coded port number. Replace the number with any
	// unique number > 1024.
  uint16_t port {21234};

  int message_count {0}; // count # of messages we received.

  constexpr std::size_t buffer_size {2048};

	if (argc > 1)
	{
		port = std::atoi(argv[1]);
	}

	std::array<char, buffer_size> send_buffer; // receive buffer

	// Create a UDP socket.
	UdpSocket socket {};

	// Bind the socket to any valid IP address and a specific port.

	auto bind_result = bind_to_any_ip_address(socket, port);
	std::cout << "Bind had errors: " << static_cast<bool>(bind_result.first) << '\n';
	std::cout << "Binded Sender address: "; 

	ToBytes((*bind_result.second).sin_port).increasing_addresses_print();
	std::cout << "\n";
	ToBytes((*bind_result.second).sin_addr.s_addr).increasing_addresses_print();
	std::cout << "\n";
	ToBytes((*bind_result.second).sin_port).increasing_addresses_print();
	std::cout << "\n";

	//Receiver receiver {port};

	ReceivingOn receive_on {};
	auto received_from = receive_on(socket);

	SendTo<buffer_size> sender_to {};

	for (;;)
	{
		std::cout << "Waiting on port " << port << "\n";
		
		// TODO: fix Receiver.
		/*
		auto receipt = receiver.receiving_data<buffer_size>();
		
		if (receipt.second)
		{
			std::string received_message (
				(*receipt.second).buffer_.cbegin(), (*receipt.second).buffer_.cend());

			std::cout << "Received message: \"" << received_message << "\" (" <<
				(*receipt.second).received_bytes_ << " bytes)\n";
		}
		else
		{
			std::cout << "Un oh - something went wrong!\n";
		}
		*/

		auto result = received_from();

		if (!static_cast<bool>(result.first))
		{
			std::cout << "received message: \"" << (*result.second).buffer_.data() <<
				"\"" << (*result.second).received_bytes_ << " bytes)\n";
		}
		else
		{
			std::cout << "Uh oh - something went wrong!\n";
		}

		std::string message {"ack "};
		message += std::to_string(message_count++);
		std::copy(message.begin(), message.end(), send_buffer.data());

		std::cout << "Sending response \"" << send_buffer.data() << "\"\n";

		auto send_message = sender_to((*result.second).sender_address_);
		auto send_result = send_message(send_buffer)(socket);

		if (static_cast<bool>(send_result.first))
		{
			std::cerr << "sendto error" << "\n";
		}

	}	
	// Never exits.
}
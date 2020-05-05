//------------------------------------------------------------------------------
/// \file MainUDP-Receiver.cpp
/// \author Ernest Yeung
/// \brief Main file to be a UDP receiver.
///
/// \details int main() is needed.
/// \ref
///-----------------------------------------------------------------------------
#include "IPC/Sockets/UDP/Receiver.h"

#include <cstdlib> // std::atoi
#include <iostream>
#include <string>
#include <utility>

using IPC::Sockets::UDP::Receiver;

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

	Receiver receiver {port};

	for (;;)
	{
		std::cout << "Waiting on port " << port << "\n";
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
		

	}	
	// Never exits.
}
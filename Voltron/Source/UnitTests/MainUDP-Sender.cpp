//------------------------------------------------------------------------------
/// \file MainUDP-Sender.cpp
/// \author Ernest Yeung
/// \brief Main file to be UDP Sender.
///
/// \details int main() is needed.
/// \ref https://www.cs.rutgers.edu/~pxk/417/notes/sockets/demo-udp-03.html
///-----------------------------------------------------------------------------
#include "IPC/Sockets/Socket.h"
#include "IPC/Sockets/UDP/CreateSocket.h"

#include "Utilities/ToBytes.h"

using IPC::Sockets::Socket;
using IPC::Sockets::UDP::UdpSocket;
using IPC::Sockets::UDP::bind_to_any_ip_address;
using Utilities::ToBytes;

#include "IPC/Sockets/UDP/Sender.h"

#include <algorithm>
#include <array>
#include <iostream>
#include <string>
#include <utility>

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <netdb.h>
#include <sys/socket.h>

#include <unistd.h> // ::close

using IPC::Sockets::UDP::Sender;

int main(int argc, char* argv[])
{
	// Arbitrarily chosen, hard-coded port number. Replace the number with any
	// unique number > 1024.
/*  uint16_t destination_port {21234};

  // Change this to use a different server.
  const std::string server {"127.0.0.1"};

  constexpr int number_of_messages {5};

  constexpr std::size_t buffer_size {2048};

  std::array<char, buffer_size> buffer;

	Sender sender {destination_port};

	auto send_on_packet = sender(server);

	for (int i {0}; i < number_of_messages; ++i)
	{
		std::cout << "Sending packet " << i << " to " << server << " port " <<
			destination_port << "\n";

		const std::string to_send {"This is packet " + std::to_string(i)};

		std::copy(to_send.begin(), to_send.end(), buffer.begin());

		auto send_on_packet_result = send_on_packet(buffer);
	}	
*/ 
	constexpr std::size_t BUFLEN {2048};
	constexpr int MSGS {5}; // number of messages to send
	constexpr int SERVICE_PORT {21234};

	struct sockaddr_in myaddr, remaddr;
	int fd, i, slen=sizeof(remaddr);
	char *server = "127.0.0.1";	/* change this to use a different server */
	char buf[BUFLEN];

	/* create a socket */

//	Socket socket {AF_INET, SOCK_DGRAM};

	UdpSocket socket {};

	//if ((fd=socket(AF_INET, SOCK_DGRAM, 0))==-1)
		//printf("socket created\n");

	/* bind it to all local addresses and pick any port number */

	auto bind_result = bind_to_any_ip_address(socket, 0);
	std::cout << "Bind had errors: " << static_cast<bool>(bind_result.first) << '\n';
	std::cout << "Binded Sender address: "; 

	ToBytes((*bind_result.second).sin_port).increasing_addresses_print();
	std::cout << "\n";
	ToBytes((*bind_result.second).sin_addr.s_addr).increasing_addresses_print();
	std::cout << "\n";
	ToBytes((*bind_result.second).sin_port).increasing_addresses_print();
	std::cout << "\n";

	/*
	memset((char *)&myaddr, 0, sizeof(myaddr));
	myaddr.sin_family = AF_INET;
	myaddr.sin_addr.s_addr = htonl(INADDR_ANY);
	myaddr.sin_port = htons(0);

	if (bind(socket.fd(), (struct sockaddr *)&myaddr, sizeof(myaddr)) < 0) {
		perror("bind failed");
		return 0;
	}       
	*/

	/* now define remaddr, the address to whom we want to send messages */
	/* For convenience, the host address is expressed as a numeric IP address */
	/* that we will convert to a binary format via inet_aton */

	memset((char *) &remaddr, 0, sizeof(remaddr));
	remaddr.sin_family = AF_INET;
	remaddr.sin_port = htons(SERVICE_PORT);
	if (inet_aton(server, &remaddr.sin_addr)==0) {
		fprintf(stderr, "inet_aton() failed\n");
		exit(1);
	}

	/* now let's send the messages */

	for (i=0; i < MSGS; i++) {
		printf(":Sending packet %d to %s port %d\n", i, server, SERVICE_PORT);
		sprintf(buf, "This is packet %d", i);
		if (sendto(socket.fd(), buf, strlen(buf), 0, (struct sockaddr *)&remaddr, slen)==-1)
			perror("sendto");
	}
	close(fd);
	return 0;
}
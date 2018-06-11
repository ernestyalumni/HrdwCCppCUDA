//------------------------------------------------------------------------------
/// \file send_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief A simple UDP client; Sends UDP messages. 
/// \ref Paul Krzyzanowski     
/// \details Sends a sequence of messages (the number of messages is defined in
///   MSGS). The messages are sent to a port defined in SERVICE_PORT.
/// \copyright If you find this code useful, feel free to donate directly and
/// easily at this direct PayPal link: 
///
/// https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
/// 
/// which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.  
/// Otherwise, I receive emails and messages on how all my (free) material on
/// physics, math, and engineering have helped students with their studies, and
/// I know what it's like to not have money as a student, but love physics (or 
/// math, sciences, etc.), so I am committed to keeping all my material 
/// open-source and free, whether or not sufficiently crowdfunded, under the 
/// open-source MIT license: feel free to copy, edit, paste, make your own
/// versions, share, use as you wish.    
/// Peace out, never give up! -EY
//------------------------------------------------------------------------------
/// COMPILATION TIPS:
///  g++ -std=c++14 Socket_main.cpp -o Socket_main
//------------------------------------------------------------------------------
#include "../Socket.h" // May need to change this depending upon file directory
#include "UdpSocket.h"
#include "port.h"

#include <array>
#include <cstddef> // std::size_t, for std::array and UdpSocket
#include <arpa/inet.h> // inet_aton
#include <iostream>

using Sockets::Socket;
using Sockets::Udp::DestinationSocketAddress;
using Sockets::Udp::SERVICE_PORT;
using Sockets::Udp::UdpSocket;

int main()
{
  constexpr const unsigned int MSGS {5}; // number of messages to send
  constexpr const std::size_t BUFLEN {2048};

  // change this to use a different server
  constexpr const std::array<char, 10> server {"127.0.0.1"}; 

  // Create a socket.
  UdpSocket<BUFLEN> udp_socket;

  // Bind it to all local addresses and pick any port number
  udp_socket.bind();

  // Now define remaddr, the address to whom we want to send messages.
  // For convenience, the host address is expressed as a numeric IP address
  // that we will convert to a binary format via inet_aton.
  DestinationSocketAddress destination_socket_address {SERVICE_PORT};

//  ::inet_aton(server.data(), )

  destination_socket_address.Ipv4address_to_network_byte_order(server.data());

  // Now let's send the messages
  for (int i {0}; i < MSGS; i++)
  {
    std::cout << "Sending packet " << i << " to " << server.data() <<
      " port " << SERVICE_PORT << '\n';
    std::string string_message {
      "This is packet " + std::to_string(i)
    };
    udp_socket.set_buffer(string_message);
    udp_socket.sendto(destination_socket_address);
  }  

}


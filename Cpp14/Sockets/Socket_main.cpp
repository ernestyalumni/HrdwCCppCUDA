//------------------------------------------------------------------------------
/// \file Socket_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  A socket as RAII 
/// \ref      
/// \details Using RAII for socket. 
/// \copyright If you find this code useful, feel free to donate directly and easily at 
/// this direct PayPal link: 
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
#include "Socket.h"

#include <iostream>

using Sockets::CommonDomains;
using Sockets::Socket;
using Sockets::SocketAddress;

class TestSocket : public Socket
{
  public:
    using Socket::fd;
};

int main()
{
  // SocketDefaultConstructs
  TestSocket test_socket;
  std::cout << " test_socket.fd() : " << test_socket.fd() << '\n';

  // CommonDomainsRepresentsSocketCommunicationDomains
  std::cout << " CommonDomains::unix : " << 
    static_cast<int>(CommonDomains::unix) << '\n'; // 1
  std::cout << " CommonDomains::local : " << 
    static_cast<int>(CommonDomains::local) << '\n'; // 1
  std::cout << " CommonDomains::ipv4_internet : " << 
    static_cast<int>(CommonDomains::ipv4_internet) << '\n'; // 2 
  std::cout << " CommonDomains::packet : " << 
    static_cast<int>(CommonDomains::packet) << '\n'; // 17

  // SocketAddressConstructsCorrectly
    SocketAddress socket_address {
      static_cast<uint32_t>(CommonDomains::ipv4_internet), 0, INADDR_ANY};

}

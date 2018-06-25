//------------------------------------------------------------------------------
/// \file TcpSocket_main.cpp
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
#include "TcpSocket.h"
#include "../Socket.h"

#include <array>
#include <cstddef> // std::size_t, for std::array
#include <iostream>
#include <sys/socket.h> // AF_INET

using Sockets::SocketAddressIn;
using Sockets::Tcp::TcpSocket;

//class TestTcpSocket : public TcpSocket<1000>
class TestTcpSocket : public TcpSocket
{
  public:

    using TcpSocket::TcpSocket;
    using TcpSocket::fd;

//    using TcpSocket<1000>::TcpSocket;
//    using TcpSocket<1000>::fd;
};

int main()
{

  // TcpSocketConstructs
  TestTcpSocket test_tcp_socket {};

  std::cout << " test_tcp_socket.fd() : " << test_tcp_socket.fd() << '\n';

  // SocketAddressConstructsForTCPParameters
  SocketAddressIn test_tcp_socket_address {AF_INET, 8888, INADDR_ANY};

  // TcpSocketConstructsWithSocketAddress
  TestTcpSocket test_tcp_socket1 {test_tcp_socket_address}; 

  // TcpSocketListens
  test_tcp_socket.bind();
  test_tcp_socket.get_socket_name();
  test_tcp_socket.listen(test_tcp_socket.fd(), 3);

  // TcpSocketAcceptsIncomingConnection
  std::cout << "\n Waiting for incoming connections... \n";

  // accept connection from an incoming client 
  SocketAddressIn client_socket_address_in;

  test_tcp_socket.accept(test_tcp_socket.fd(), client_socket_address_in);

  // TcpSocketReceivesFromItsOwnFd
  
  
  
}

#if 0
  std::cout << " test_tcp_socket.domain() : " << test_tcp_socket.domain() <<
    "  (test_tcp_socket.domain() == AF_INET) : " <<
    (test_tcp_socket.domain() == AF_INET) << '\n';

  std::cout << " test_tcp_socket.type() : " << test_tcp_socket.type() <<
    "  (test_tcp_socket.type() == SOCK_STREAM) : " <<
    (test_tcp_socket.type() == SOCK_STREAM) << '\n';

  std::cout << " test_tcp_socket.protocol() : " << test_tcp_socket.protocol() <<
    "  (test_tcp_socket.protocol() == 0) : " <<
    (test_tcp_socket.protocol() == 0) << '\n';
#endif 

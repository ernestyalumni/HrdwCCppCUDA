//------------------------------------------------------------------------------
/// \file UdpSocket_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  A UDP socket as RAII 
/// \ref      
/// \details Using RAII for UDP socket. 
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
#include <sys/socket.h> // ::socket, ::sendto

#include <arpa/inet.h> // ::htonl, ::htons, ::ntohl, ::ntohs

#include <netinet/in.h> // ::sockaddr_in
#include <unistd.h> // ::close

#include <array>
#include <memory>

using Sockets::SocketAddress;

std::array<char, 12> hello_world {"hello world"};

int main()
{
  int sockfd {::socket(AF_INET, SOCK_DGRAM | SOCK_NONBLOCK, 0)};
  std::cout << " sockfd : " << sockfd << '\n';


  constexpr const uint16_t port {58080};

  ::sockaddr_in sock_address {AF_INET, ::htons(port), ::htonl(INADDR_ANY) };
  std::cout << " sock_address.sin_family : " << sock_address.sin_family << ' ' <<
    " sock_address.sin_port : " << sock_address.sin_port << ' ' << 
    " sock_address.sin_addr.s_addr : " << sock_address.sin_addr.s_addr <<
    " ::ntohs(sock_address.sin_port) : " << ::ntohs(sock_address.sin_port) << 
    ' ' << " ::ntohl(sock_address.sin_addr.s_addr : " << 
    ::ntohs(sock_address.sin_addr.s_addr) << ' ' << '\n';

  std::unique_ptr<::sockaddr_in> sock_address_uptr {
    std::make_unique<::sockaddr_in>(sock_address)
  };

  int bind_result {::bind(sockfd, 
    reinterpret_cast<::sockaddr*>(sock_address_uptr.get()), sizeof(::sockaddr_in)
    )};

  std::cout << " bind_result : " << bind_result << '\n'; // 0 for success

  ssize_t sendto_result {::sendto(sockfd, hello_world.data(),
    hello_world.size(), 0,
    reinterpret_cast<::sockaddr*>(sock_address_uptr.get()),
    sizeof(::sockaddr_in))};

  std::cout << " sendto_result : " << sendto_result << '\n';


  ::sockaddr_in socket_address_name;
  std::unique_ptr<::sockaddr_in> socket_address_name_uptr {
    std::make_unique<::sockaddr_in>(socket_address_name)};


  socklen_t sockaddr_in_length {sizeof(::sockaddr_in)};
  std::unique_ptr<socklen_t> sockaddr_in_length_uptr {
    std::make_unique<socklen_t>(sockaddr_in_length)
  };

  int getsockname_results {
    ::getsockname(sockfd, 
      reinterpret_cast<::sockaddr*>(socket_address_name_uptr.get()),
    sockaddr_in_length_uptr.get())
//    &sockaddr_in_length)
  };

  std::cout << " getsockname_results : " << getsockname_results << '\n'; // 0 for success

  std::cout << "socket_address_name.sin_family : " << 
    socket_address_name_uptr.get()->sin_family << 
    ' ' << " socket_address_name.sin_port : " << 
    socket_address_name_uptr.get()->sin_port << ' ' << 
    "socket_address_name.sin_addr.s_addr : " << 
    socket_address_name_uptr.get()->sin_addr.s_addr <<
    " ::ntohs(socket_address_name.sin_port) : " << 
    ::ntohs(socket_address_name_uptr.get()->sin_port) << 
    ' ' << " ::ntohl(socket_address_name.sin_addr.s_addr : " << 
    ::ntohs(socket_address_name_uptr.get()->sin_addr.s_addr) << ' ' << '\n';

  

  int close_result {::close(sockfd)};
  std::cout << " close_result : " << close_result << std::endl;

}

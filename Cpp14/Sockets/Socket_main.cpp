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
#include <memory>
#include <netinet/in.h> // ::sockaddr_in
#include <stdexcept>
#include <unistd.h> // ::close
#include <string>

using Sockets::CommonDomains;
using Sockets::Socket;
using Sockets::SocketAddress;
using Sockets::SocketAddressIn;
using Sockets::SocketV2;

class TestSocket : public Socket
{
  public:

    using Socket::fd;
};

class TestSocketV2 : public SocketV2
{
  public:

    using SocketV2::SocketV2; 
    using SocketV2::fd;
    using SocketV2::socket_address_in;
};

int main()
{

  // ::sockaddr_inConstructs
  ::sockaddr_in test_sockaddr_in {AF_INET, ::htons(8888), ::htonl(INADDR_ANY)};
  std::cout << " test_sockaddr_in.sin_family : " << 
    test_sockaddr_in.sin_family << 
    " (test_sockaddr_in.sin_family == AF_INET): " << 
    (test_sockaddr_in.sin_family == AF_INET) << '\n';

  std::cout << " test_sockaddr_in.sin_port : " << 
    test_sockaddr_in.sin_port << 
    " ::ntohs(test_sockaddr_in.sin_port): " << 
    ::ntohs(test_sockaddr_in.sin_port) << '\n';

  std::cout << " test_sockaddr_in.sin_addr.s_addr : " << 
    test_sockaddr_in.sin_addr.s_addr << 
    " ::ntohl(test_sockaddr_in.sin_addr.s_addr): " << 
    ::ntohl(test_sockaddr_in.sin_addr.s_addr) << '\n';

  // SocketAddressInConstructs
  SocketAddressIn test_tcp_socket_address_in;

  // Construct with an actual port number.
  SocketAddressIn test_tcp_socket_address_in1 {AF_INET, 8888, INADDR_ANY};
  std::cout << "\n SocketAddressConstructs \n";

  std::cout << " test_tcp_socket_address_in.sin_family : " << 
    test_tcp_socket_address_in.sin_family << 
    " (test_tcp_socket_address_in.sin_family == AF_INET): " << 
    (test_tcp_socket_address_in.sin_family == AF_INET) << '\n';

  std::cout << " test_tcp_socket_address_in.sin_port : " << 
    test_tcp_socket_address_in.sin_port << 
    " ::ntohs(test_tcp_socket_address_in.sin_port): " << 
    ::ntohs(test_tcp_socket_address_in.sin_port) << '\n';

  std::cout << " test_tcp_socket_address_in.sin_addr.s_addr : " << 
    test_tcp_socket_address_in.sin_addr.s_addr << 
    " ::ntohl(test_tcp_socket_address_in.sin_addr.s_addr): " << 
    ::ntohl(test_tcp_socket_address_in.sin_addr.s_addr) << '\n';

  std::cout << " test_tcp_socket_address_in1.sin_family : " << 
    test_tcp_socket_address_in1.sin_family << 
    " (test_tcp_socket_address_in1.sin_family == AF_INET): " << 
    (test_tcp_socket_address_in1.sin_family == AF_INET) << '\n';

  std::cout << " test_tcp_socket_address_in1.sin_port : " << 
    test_tcp_socket_address_in1.sin_port << 
    " ::ntohs(test_tcp_socket_address_in1.sin_port): " << 
    ::ntohs(test_tcp_socket_address_in1.sin_port) << '\n';

  std::cout << " test_tcp_socket_address_in1.sin_addr.s_addr : " << 
    test_tcp_socket_address_in1.sin_addr.s_addr << 
    " ::ntohl(test_tcp_socket_address_in1.sin_addr.s_addr): " << 
    ::ntohl(test_tcp_socket_address_in1.sin_addr.s_addr) << '\n';

  // SocketDefaultConstructs
  std::cout << "\n SocketConstructs \n";
  TestSocket test_socket;
  std::cout << " test_socket.fd() : " << test_socket.fd() << '\n';

  // ::bindBindsSocketAddressInWithSocket
  std::cout << "\n ::bindBindsSocketAddressInWithSocket \n";

  // WORKS but DOES NOT BIND
  std::unique_ptr<::sockaddr> test_tcp_socket_address_uptr {
    std::make_unique<::sockaddr>(
      reinterpret_cast<::sockaddr&>(test_tcp_socket_address_in))
  };

  // this WORKS but BIND ERROR
//  std::shared_ptr<SocketAddressIn> test_tcp_socket_address_in_uptr {
  //  std::make_shared<SocketAddressIn>(test_tcp_socket_address_in)
 // };

  // std::reinterpret_pointer_cast only in C++17
//  std::shared_ptr<::sockaddr> test_tcp_socket_address_uptr {
  //  std::reinterpret_pointer_cast<::sockaddr, SocketAddressIn>(
    //  test_tcp_socket_address_in_uptr)
//  };

  // this compiles but BIND error
//  std::unique_ptr<::sockaddr> test_tcp_socket_address_uptr (
  //  std::move(reinterpret_cast<::sockaddr*>(&test_tcp_socket_address_in)));

  // this does NOT WORK
//  std::unique_ptr<::sockaddr> test_tcp_socket_address_uptr {
  //  reinterpret_cast<std::unique<::sockaddr>>(test_tcp_socket_address_in_uptr)
  //};
  
  // compiles but BIND error, invalid pointer
//  std::unique_ptr<::sockaddr> test_tcp_socket_address_uptr {
  //  reinterpret_cast<::sockaddr*>(&test_tcp_socket_address_in)
  //};

  const int bind_result {
    ::bind(
      test_socket.fd(),
      test_tcp_socket_address_uptr.get(),
      test_tcp_socket_address_in.size())
  };
  std::cout << " bind_result : " << bind_result << '\n';

  SocketAddressIn test_socket_address_in;
  TestSocket test_socket1;
  std::cout << " test_socket1.fd() : " << test_socket1.fd() << '\n';

  sockaddr* test_socket_address_ptr {
    reinterpret_cast<::sockaddr*>(&test_socket_address_in)
  };

  const int bind_result1 {
    ::bind(
      test_socket1.fd(),
      test_socket_address_ptr,
      test_socket_address_in.size())
  };
  std::cout << " bind_result1 : " << bind_result1 << '\n';

  SocketAddressIn test_socket_address_in2;
  TestSocket test_socket2;
  std::cout << " test_socket2.fd() : " << test_socket2.fd() << '\n';

  const int bind_result2 {
    ::bind(
      test_socket2.fd(),
      test_socket_address_in2.to_sockaddr(),
      test_socket_address_in2.size())
  };
  std::cout << " bind_result2 : " << bind_result2 << '\n';

  // ::getsocknameGetsActualPort
  std::cout << "\n ::getsocknameGetsActualPort \n";


  // this DOES NOT WORK
//  std::unique_ptr<::sockaddr> test_tcp_socket_address_uptr {
  //  std::make_unique<::sockaddr>(
    //  reinterpret_cast<::sockaddr*>(&test_tcp_socket_address_in))
  //};

  // this WORKS
//  ::sockaddr* test_tcp_socket_address_reference {
//    reinterpret_cast<::sockaddr*>(&test_tcp_socket_address_in)
//  };

  // this WORKS
//  ::sockaddr& test_tcp_socket_address_reference {
  //  reinterpret_cast<::sockaddr&>(test_tcp_socket_address_in)
  //};

  std::unique_ptr<socklen_t> socket_length_uptr {
    std::make_unique<socklen_t>(sizeof(::sockaddr_in))};

  std::cout << " *(socket_length_uptr.get()) : " <<
    *(socket_length_uptr.get()) << '\n';

  const int get_socket_name_result {
    ::getsockname(test_socket.fd(), 
    test_tcp_socket_address_uptr.get(),
    socket_length_uptr.get())};
  std::cout << " get_socket_name_result : " << get_socket_name_result << '\n';
  std::cout << " test_tcp_socket_address_uptr.get() : " <<
    test_tcp_socket_address_uptr.get() <<
    " *test_tcp_socket_address_uptr.get() " << 
    std::string(test_tcp_socket_address_uptr.get()->sa_data) << '\n';
  for (int i {0}; i < 12; i++)
  {
    std::cout << test_tcp_socket_address_uptr.get()->sa_data[i] << ' ';
  }

  std::cout << " test_tcp_socket_address_in.sin_family : " << 
    test_tcp_socket_address_in.sin_family << 
    " (test_tcp_socket_address_in.sin_family == AF_INET): " << 
    (test_tcp_socket_address_in.sin_family == AF_INET) << '\n';

  std::cout << " test_tcp_socket_address_in.sin_port : " << 
    test_tcp_socket_address_in.sin_port << 
    " ::ntohs(test_tcp_socket_address_in.sin_port): " << 
    ::ntohs(test_tcp_socket_address_in.sin_port) << '\n';

  std::cout << " test_tcp_socket_address_in.sin_addr.s_addr : " << 
    test_tcp_socket_address_in.sin_addr.s_addr << 
    " ::ntohl(test_tcp_socket_address_in.sin_addr.s_addr): " << 
    ::ntohl(test_tcp_socket_address_in.sin_addr.s_addr) << '\n';

  socklen_t socket_length {sizeof(::sockaddr_in)};

  const int get_socket_name_result1 {
    ::getsockname(test_socket1.fd(), 
      test_socket_address_ptr,
      &socket_length)};
  std::cout << " get_socket_name_result1 : " << get_socket_name_result1 << '\n';

  std::cout << " test_socket_address_in.sin_family : " << 
    test_socket_address_in.sin_family << 
    " (test_socket_address_in.sin_family == AF_INET): " << 
    (test_socket_address_in.sin_family == AF_INET) << '\n';

  std::cout << " test_socket_address_in.sin_port : " << 
    test_socket_address_in.sin_port << 
    " ::ntohs(test_socket_address_in.sin_port): " << 
    ::ntohs(test_socket_address_in.sin_port) << '\n';

  std::cout << " test_socket_address_in.sin_addr.s_addr : " << 
    test_socket_address_in.sin_addr.s_addr << 
    " ::ntohl(test_socket_address_in.sin_addr.s_addr): " << 
    ::ntohl(test_socket_address_in.sin_addr.s_addr) << '\n';

  const int get_socket_name_result2 {
    ::getsockname(test_socket2.fd(), 
      test_socket_address_in2.to_sockaddr(),
      &socket_length)};
  std::cout << " get_socket_name_result2 : " << get_socket_name_result1 << '\n';

  std::cout << " test_socket_address_in2.sin_family : " << 
    test_socket_address_in2.sin_family << 
    " (test_socket_address_in2.sin_family == AF_INET): " << 
    (test_socket_address_in2.sin_family == AF_INET) << '\n';

  std::cout << " test_socket_address_in2.sin_port : " << 
    test_socket_address_in2.sin_port << 
    " ::ntohs(test_socket_address_in2.sin_port): " << 
    ::ntohs(test_socket_address_in2.sin_port) << '\n';

  std::cout << " test_socket_address_in2.sin_addr.s_addr : " << 
    test_socket_address_in2.sin_addr.s_addr << 
    " ::ntohl(test_socket_address_in2.sin_addr.s_addr): " << 
    ::ntohl(test_socket_address_in2.sin_addr.s_addr) << '\n';

  // CommonDomainsRepresentsSocketCommunicationDomains
  std::cout << "\n CommonDomainsRepresentsSocketCommunicationDomains \n";  
  std::cout << " CommonDomains::unix : " << 
    static_cast<int>(CommonDomains::unix) << '\n'; // 1
  std::cout << " CommonDomains::local : " << 
    static_cast<int>(CommonDomains::local) << '\n'; // 1
  std::cout << " CommonDomains::ipv4_internet : " << 
    static_cast<int>(CommonDomains::ipv4_internet) << '\n'; // 2 
  std::cout << " CommonDomains::packet : " << 
    static_cast<int>(CommonDomains::packet) << '\n'; // 17

  // SocketAddressConstructsCorrectly
    std::cout << "Create a SocketAddress. \n";
    SocketAddress socket_address {
      static_cast<uint32_t>(CommonDomains::ipv4_internet), 0, INADDR_ANY};

    std::cout << " sin_family : " << 
      socket_address.get_sockaddr_in_uptr()->sin_family << '\n'; // 2

    std::cout << " sin_port : " << 
      socket_address.get_sockaddr_in_uptr()->sin_port << '\n'; // 0

    std::cout << " sin_addr.s_addr : " << 
      socket_address.get_sockaddr_in_uptr()->sin_addr.s_addr << '\n'; // 0

  // TestSocketBinds
  try
  {
    test_socket.bind();
  }
  catch (const std::runtime_error& e)
  {
    std::cout << " runtime error was caught, with message : '" << e.what() << 
      " \n";
  }

  // TestSocketCanGetSocketName
  try
  {
    const int get_socket_name_result {test_socket.get_socket_name()};
    std::cout << " get_socket_name_result : " << get_socket_name_result << '\n';
  }
  catch (const std::runtime_error& e)
  {
    std::cout << " runtime error was caught, with message : '" << e.what() << 
      " \n";    
  }

  // TestSocketFdAccessorCanBeClosedExternally
  ::close(test_socket.fd());

  // TestSocketV2ConstructsBindsAndGetsSocketName
  std::cout << "\n TestSocketV2ConstructsBindsAndGetsSocketName \n";

  TestSocketV2 test_udp_socket_v2 {AF_INET, SOCK_DGRAM};
  test_udp_socket_v2.bind();
  test_udp_socket_v2.get_socket_name();

  std::cout << " test_udp_socket_v2.socket_address_in().sin_family : " << 
    test_udp_socket_v2.socket_address_in().sin_family << 
    " (test_udp_socket_v2.socket_address_in().sin_family == AF_INET): " << 
    (test_udp_socket_v2.socket_address_in().sin_family == AF_INET) << '\n';

  std::cout << " test_udp_socket_v2.socket_address_in().sin_port : " << 
    test_udp_socket_v2.socket_address_in().sin_port << 
    " ::ntohs(test_udp_socket_v2.socket_address_in().sin_port): " << 
    ::ntohs(test_udp_socket_v2.socket_address_in().sin_port) << '\n';

  std::cout << " test_udp_socket_v2.socket_address_in().sin_addr.s_addr : " << 
    test_udp_socket_v2.socket_address_in().sin_addr.s_addr << 
    " ::ntohl(test_udp_socket_v2.socket_address_in().sin_addr.s_addr): " << 
    ::ntohl(test_udp_socket_v2.socket_address_in().sin_addr.s_addr) << '\n';

  

}

//------------------------------------------------------------------------------
/// \file Udp_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  A Udp sockets as RAII main driver file.
/// \ref      
/// \details Using RAII for Udp socket. 
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
///  g++ -std=c++14 Udp_main.cpp -o Udp_main
//------------------------------------------------------------------------------
#include "Udp.h"
#include "Socket.h"

#include <iostream>

using Sockets::AllSpecialAddresses;
using Sockets::Udp::UdpSocket;

class TestUdpSocket : public UdpSocket
{
  public:

    using UdpSocket::UdpSocket;
    using UdpSocket::fd;
    using UdpSocket::socket_address_in;
    using UdpSocket::get_socket_length_ptr;
    using UdpSocket::close;
};

int main()
{

  //----------------------------------------------------------------------------
  //----------------------------------------------------------------------------
  /// TestUdpSocketConstructsAndBindsToEphemeralPort
  TestUdpSocket test_udp_socket {
    0,
    static_cast<uint32_t>(AllSpecialAddresses::loopback)};  

  const int get_socket_name_result {test_udp_socket.get_socket_name()};
  std::cout << " get_socket_name_result : " << get_socket_name_result << '\n';

  std::cout << " \n test_udp_socket.socket_address_in().sin_port : " <<
    test_udp_socket.socket_address_in().sin_port << '\n';

}
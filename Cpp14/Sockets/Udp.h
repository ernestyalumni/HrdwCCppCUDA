//------------------------------------------------------------------------------
/// \file Udp.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Udp sockets as class template RAII 
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
#ifndef _UDP_H_
#define _UDP_H_

#include "Socket.h"

// \ref http://pubs.opengroup.org/onlinepubs/7908799/xsh/unistd.h.html
// \ref http://pubs.opengroup.org/onlinepubs/7908799/xns/netinetin.h.html
#include <arpa/inet.h> // htonl, htons
#include <cerrno> // errno
#include <cstring> // strerror
#include <iostream>
#include <memory>
#include <netinet/in.h> // ::sockaddr_in
#include <stdexcept> // std::runtime_error
#include <sys/socket.h> // AF_INET,AF_UNIX,AF_LOCAL,etc.; communication domains
#include <system_error>
#include <unistd.h> // ::close, ::sleep, ::unlink

using Sockets::AllDomains;
using Sockets::AllTypes;
using Sockets::AllSpecialAddresses;
using Sockets::SocketAddressIn;
using Sockets::Socket;

namespace Sockets
{

namespace Udp
{

using UdpSocket = Socket<
  AllDomains::ipv4_internet,
  static_cast<int>(AllTypes::datagram),
  IPPROTO_UDP>;

} // namespace Udp

} // namespace Sockets

#endif 
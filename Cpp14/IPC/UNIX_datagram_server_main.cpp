//------------------------------------------------------------------------------
/// \file UNIX_datagram_server_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  The following example shows how to create, bind, and listen on a
/// datagram socket in UNIX (AF_UNIX) domain, and accept connections.
/// \ref https://github.com/skuhl/sys-prog-examples/blob/master/simple-examples/unix-dgram-server.c    
/// \details 
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
///  g++ -I ../ -I ./Sockets/ --std=c++17 ./Sockets/Socket.cpp ../Utilities/ErrorHandling.cpp ../Utilities/Errno.cpp UNIX_datagram_server_main.cpp -o UNIX_datagram_server_main
//------------------------------------------------------------------------------
#include "Socket.h"
#include "Utilities/casts.h" // get_underlying

#include <iostream>
#include <string>

using IPC::Sockets::Domains;
using IPC::Sockets::Socket;
using IPC::Sockets::Types;
using Utilities::get_underlying_value;

int main()
{
  const std::string socket_path_server {"unix-dgram-server.temp"};
  const std::string socket_path_client {"unix-dgram-client.temp"};

  // A call to ::socket() with proper arguments creates the Unix socket.

  // SOCK_DGRAM second argument tells ::socket() a datagram socket.
  // The number of ::read() or ::recv() must match number of ::send() or
  // ::write().
  // If you read with a size smaller than size of packet, you won't receive
  // entire message.
  Socket socket {
    Domains::unix,
    get_underlying_value<Types>(Types::datagram),
    0};

  

}
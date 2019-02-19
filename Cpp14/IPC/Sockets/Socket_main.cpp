//------------------------------------------------------------------------------
/// \file Socket_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Main driver file for a socket as RAII 
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
///  g++ -I ../../ -std=c++14 Socket.cpp Socket_main.cpp ../../Utilities/ErrorHandling.cpp ../../Utilities/Errno.cpp -o Socket_main
//------------------------------------------------------------------------------
#include "Socket.h"

#include "Utilities/casts.h" // get_underlying

#include <iostream>

using IPC::Sockets::Domains;
using IPC::Sockets::Socket;
using IPC::Sockets::Types;
using Utilities::get_underlying_value;

int main()
{
  // Domains
  {
    std::cout << 
      "unix : " << get_underlying_value<Domains>(Domains::unix) << '\n' <<
      "local : " << get_underlying_value<Domains>(Domains::local) << '\n' <<
      "ipv4_internet : " <<
        get_underlying_value<Domains>(Domains::ipv4_internet) << '\n' <<
      "ipv6_internet : " <<
        get_underlying_value<Domains>(Domains::ipv6_internet) << '\n' <<
      "ipx_Novell : " <<
        get_underlying_value<Domains>(Domains::ipx_Novell) << '\n' <<
      "kernel_user_interface : " <<
        get_underlying_value<Domains>(Domains::kernel_user_interface) << '\n' <<
      "itu_t_x25 : " <<
        get_underlying_value<Domains>(Domains::itu_t_x25) << '\n' <<
      "amateur_radio : " <<
        get_underlying_value<Domains>(Domains::amateur_radio) << '\n' <<
      "raw_ATM_PVCs : " <<
        get_underlying_value<Domains>(Domains::raw_ATM_PVCs) << '\n' <<
      "appletalk : " <<
        get_underlying_value<Domains>(Domains::appletalk) <<
      '\n';
  }

  // Types
  {
    std::cout << "\n Types \n";

    std::cout << 
      "stream : " << get_underlying_value<Types>(Types::stream) << '\n' <<
      "datagram : " << get_underlying_value<Types>(Types::datagram) << '\n' <<
      "sequenced_packets : " << 
        get_underlying_value<Types>(Types::sequenced_packets) << '\n' <<
      "raw : " << get_underlying_value<Types>(Types::raw) << '\n' <<
      "reliable_datagram_layer : " <<
        get_underlying_value<Types>(Types::reliable_datagram_layer) << '\n' <<
      "nonblocking : " << get_underlying_value<Types>(Types::nonblocking) <<
      "\n close_on_exec : " << get_underlying_value<Types>(Types::close_on_exec) <<
        "\n";
  }

  // SocketDefaultConstructs
  {
    std::cout << "\n SocketConstructs \n";
    {
      Socket unix_socket {Domains::unix,
        get_underlying_value<Types>(Types::stream) ||
          get_underlying_value<Types>(Types::nonblocking),
        0};

      std::cout << unix_socket << '\n';
    }

    {
      Socket local_socket {Domains::local,
        get_underlying_value<Types>(Types::datagram) ||
          get_underlying_value<Types>(Types::nonblocking),
        0};

      std::cout << local_socket << '\n';
    }

    {
      Socket ipv6_socket {Domains::ipv6_internet,
        get_underlying_value<Types>(Types::reliable_datagram_layer) ||
          get_underlying_value<Types>(Types::nonblocking),
        0};

      std::cout << ipv6_socket << '\n';
    }

    {
      Socket unix_socket {Domains::ipv4_internet,
        get_underlying_value<Types>(Types::sequenced_packets) ||
          get_underlying_value<Types>(Types::nonblocking),
        0};

      std::cout << unix_socket << '\n';
    }

    // \ref http://man7.org/linux/man-pages/man7/netlink.7.html
    {
      Socket kernel_socket {Domains::kernel_user_interface,
        get_underlying_value<Types>(Types::raw),
        0};

      std::cout << kernel_socket << '\n';
    }
  }
}
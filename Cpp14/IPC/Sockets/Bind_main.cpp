//------------------------------------------------------------------------------
/// \file Bind_main.cpp
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
///  g++ -I ../../ -std=c++14 Socket.cpp Bind_main.cpp ../../Utilities/ErrorHandling.cpp ../../Utilities/Errno.cpp -o Bind_main
//------------------------------------------------------------------------------
#include "Bind.h"
#include "Socket.h"
#include "Utilities/casts.h" // get_underlying

#include <cstring> // std::memset, std::strncpy
#include <iostream>
#include <string>
#include <sys/un.h>
#include <type_traits>
#include <unistd.h> // ::unlink

#define MY_SOCK_PATH "/somepath"

using IPC::Sockets::BindAddressFamily;
using IPC::Sockets::Domains;
using IPC::Sockets::Socket;
using IPC::Sockets::Types;
using Utilities::get_underlying_value;

int main()
{
  {
//    const std::string my_sock_path {"/somepath"};

    int sfd, cfd;
    struct ::sockaddr_un my_addr, peer_addr;
    socklen_t peer_addr_size;
#if 0
    sfd = ::socket(AF_UNIX, SOCK_STREAM, 0);
    if (sfd == -1)
    {
      std::cout << " sfd : " << sfd << '\n';
    }

    std::memset(&my_addr, 0, sizeof(struct ::sockaddr_un));
                         /* Clear structure */
    
    my_addr.sun_family = AF_UNIX;
    std::strncpy(my_addr.sun_path, MY_SOCK_PATH,
             sizeof(my_addr.sun_path) - 1);

    int bind_result = ::bind(sfd, (::sockaddr *) &my_addr,
             sizeof(::sockaddr_un));

    if (bind_result < 0)
    {
      std::cout << " bind_result : " << bind_result << '\n';
    }
#endif
  }

  // \ref http://man7.org/linux/man-pages/man2/bind.2.html
  {
    #if 0
    const std::string my_sock_path {"/somepath"};

    ::sockaddr_un my_addr;

    Socket socket {
      Domains::unix,
      get_underlying_value<Types>(Types::stream),
      0};

    if (std::is_trivially_copyable<::sockaddr_un>::value)
    {
      std::cout << "\n ::sockaddr_un is is_trivially_copyable \n";

      // Clear structure
      std::memset(&my_addr, 0, sizeof(::sockaddr_un));
    }

    my_addr.sun_family = get_underlying_value<Domains>(Domains::unix);

    std::strncpy(
      my_addr.sun_path,
      my_sock_path.c_str(),
      sizeof(my_addr.sun_path) - 1);

    BindAddressFamily<::sockaddr_un> bind_sockaddr_un {my_addr};

    bind_sockaddr_un(socket);
  #endif 
  }

  // \url https://github.com/skuhl/sys-prog-examples/blob/master/simple-examples/unix-dgram-server.c 
  {
    const std::string socket_path_server {"unix-dgram-server.temp"};
    const std::string socket_path_client {"unix-dgram-client.temp"};

    // SOCK_DGRAM tells socket() a datagram socket; the number of read() or
    // recv() must match the number of send() or write(). If you read with a
    // size smaller than the size of the packet, you won't receive the entire
    // message.

    Socket socket {
      Domains::unix,
      get_underlying_value<Types>(Types::datagram),
      0};

    // You got a socket descriptor from the call to ::socket(); now you want to
    // bind that to an address in the Unix domain. (That address is a special
    // file on disk.)
    //
    // This associates the socket descriptor "s" with the Unix socket address
    // "/home/beej/mysocket". Notice that we called ::unlink() before ::bind()
    // to remove the socket if it already exists. You'll get an EINVAL error if
    // the file is already there.      
    ::sockaddr_un local;

    local.sun_family = get_underlying_value<Domains>(Domains::unix);

    std::strcpy(local.sun_path, socket_path_server.c_str());

    ::unlink(local.sun_path);

    std::size_t len {std::strlen(local.sun_path) + sizeof(local.sun_family)};
    std::cout << " len : " << len << " std::strlen(local.sun_path) : " <<
      std::strlen(local.sun_path) << ' ' << socket_path_server.size() <<
      " sizeof(local.sun_family) : " << sizeof(local.sun_family) << '\n';

    std::cout << " sizeof(::sockaddr_un) : " << sizeof(::sockaddr_un) << '\n';

    BindAddressFamily<::sockaddr_un> bind_sockaddr_un {local};

    bind_sockaddr_un(socket, len);


    // Don't have to call ::listen() for a datagram connection.

//    for (;;)
    {
      // Don't have to accept() or wait for a connection, just wait for
      // datagrams...

//      std::cout << 
    }

  }
}
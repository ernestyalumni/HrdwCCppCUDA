//------------------------------------------------------------------------------
/// \file SocketAddress_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Wrappers and examples classes for errno and error handling.
/// \ref https://linux.die.net/man/2/epoll_ctl
/// https://en.cppreference.com/w/cpp/error/errno_macros
/// \details Event describes object linked to fd.  struct epoll_event.
/// \copyright If you find this code useful, feel free to donate directly
/// (username ernestyalumni or email address above), going directly to:
///
/// paypal.me/ernestyalumni
///
/// which won't go through a 3rd. party like indiegogo, kickstarter, patreon.
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
///   g++ --std=c++17 -I ../../ InternetAddress.cpp InternetAddress_main.cpp -o InternetAddress_main
//------------------------------------------------------------------------------
#include "InternetAddress.h"

#include "Std/TypeTraitsProperties.h"

#include <iostream>
#include <sys/socket.h> // ::sockaddr
#include <sys/un.h>
#include <netinet/ip.h> // ::sockaddr_in

using Std::PrimaryTypeTraits;
using Std::CompositeTypeTraits;

int main()
{
  //----------------------------------------------------------------------------
  /// \url http://man7.org/linux/man-pages/man2/bind.2.html
  /// \details int bind(int sockfd, const struct sockaddr *addr,
  ///   socklen_t addrlen);
  /// The actual structure passed for the addr argument will depend on the
  /// address family (AF).
  /// The sockaddr structure is defined as something like:
  /// struct ::sockaddr
  /// {
  ///   sa_family_t sa_family;
  ///   char sa_data[14];
  /// }
  /// The only purpose of this structure is to cast the structure
  /// pointer passed in addr in order to avoid compiler warnings. 
  //----------------------------------------------------------------------------
  {
    std::cout << " sanity check: size(char) : " << sizeof(char) << '\n'; // 1
    std::cout << " sizeof(::sockaddr) : " << sizeof(::sockaddr) << '\n'; // 16
    std::cout << " sizeof(sa_family_t) : " << sizeof(sa_family_t) << '\n'; // 2
    std::cout << " PrimaryTypeTraits<sa_family_t>{} : " <<
      PrimaryTypeTraits<sa_family_t>{} << '\n';
    std::cout << " CompositeTypeTraits<sa_family_t>{} : " <<
      CompositeTypeTraits<sa_family_t>{} << '\n';

    std::cout << " sizeof(socklen_t) : " << sizeof(socklen_t) << '\n'; // 2
    std::cout << " PrimaryTypeTraits<socklen_t>{} : " <<
      PrimaryTypeTraits<socklen_t>{} << '\n';
    std::cout << " CompositeTypeTraits<socklen_t>{} : " <<
      CompositeTypeTraits<socklen_t>{} << '\n';


    std::cout << " PrimaryTypeTraits<::sockaddr>{} : " <<
      PrimaryTypeTraits<::sockaddr>{} << '\n';
    std::cout << " CompositeTypeTraits<::sockaddr>{} : " <<
      CompositeTypeTraits<::sockaddr>{} << '\n';

    std::cout << " PrimaryTypeTraits<sa_family_t>{} : " <<
      PrimaryTypeTraits<sa_family_t>{} << '\n';
    std::cout << " CompositeTypeTraits<sa_family_t>{} : " <<
      CompositeTypeTraits<sa_family_t>{} << '\n';

    std::cout << " sizeof(::sockaddr_un) : " << sizeof(::sockaddr_un) << '\n'; // 16
    std::cout << " PrimaryTypeTraits<::sockaddr_un>{} : " <<
      PrimaryTypeTraits<::sockaddr_un>{} << '\n';
    std::cout << " CompositeTypeTraits<::sockaddr_un>{} : " <<
      CompositeTypeTraits<::sockaddr_un>{} << '\n';
  }

  {
    std::cout << " sizeof(in_port_t) : " << sizeof(in_port_t) << '\n'; // 2
    std::cout << " PrimaryTypeTraits<in_port_t>{} : " <<
      PrimaryTypeTraits<in_port_t>{} << '\n';
    std::cout << " CompositeTypeTraits<in_port_t>{} : " <<
      CompositeTypeTraits<in_port_t>{} << '\n';


    std::cout << " sizeof(::sockaddr_in) : " << sizeof(::sockaddr_in) << '\n'; // 16
    std::cout << " PrimaryTypeTraits<::sockaddr_in>{} : " <<
      PrimaryTypeTraits<::sockaddr_in>{} << '\n';
    std::cout << " CompositeTypeTraits<::sockaddr_in>{} : " <<
      CompositeTypeTraits<::sockaddr_in>{} << '\n';
  }
}
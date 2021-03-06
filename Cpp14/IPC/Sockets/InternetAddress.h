//------------------------------------------------------------------------------
/// \file InternetAddress.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Wrappers and classes for socket addresses.
/// \ref http://man7.org/linux/man-pages/man7/ip.7.html
/// \details Only 1 IP (Internet Protocol) socket may be bound to any given
/// local (address, port) pair.
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
///   g++ --std=c++17 -I ../../ Event.cpp Event_main.cpp -o Event_main
//------------------------------------------------------------------------------
#ifndef _IPC_SOCKETS_INTERNET_ADDRESS_H_
#define _IPC_SOCKETS_INTERNET_ADDRESS_H_

#include <netinet/ip.h>
#include <ostream>

namespace IPC
{

namespace Sockets
{

//------------------------------------------------------------------------------
/// \brief Derived class for ::sockaddr_in
/// \url http://man7.org/linux/man-pages/man7/ip.7.html
/// \details An IP socket address is defined as a combination of an IP interface
/// address and a 16-bit port number (2 bytes). The basic IP protocol does not
/// supply port numbers; they're implemented by higher level protocols like udp
/// and tcp.
/// On raw sockets sin_port is set to the IP protocol. 
/// 
/// struct sockaddr_in
/// {
///   sa_family_t sin_family; // address familly: AF_INET
///   in_port_t sin_port // port in network byte order
///   struct in_addr sin_addr; // internet address
/// };
///
/// // Internet address
/// struct in_addr
/// {
///   uint32_t s_addr; // address in network byte order
/// };
/// sin_family is always set to AF_INET. This is required; in Linux 2.2 most
/// networking functions return EINVAL when this setting is missing.
/// sin_port contains the port in network byte order. 
///   - Port numbers below 1024 are called privileged ports (or sometimes:
///   reserved ports).
/// Only a privileged process (on Linux: a process that has the
/// CAP_NET_BIND_SERVICE capability in user namespace governing its network
/// namespace) may bind to these sockets.
///   - Note that raw IPv4 protocol as such has no concept of a port; they're
/// implemented only by higher protocols like tcp, udp.
///
/// sin_addr is IP host address. s_addr member of struct in_addr contains the
/// host interface address in network byte order.
/// in_addr should be assigned 1 of the INADDR_* values (e.g. INADDR_LOOPBACK)
/// using htonl or set using the
/// inet_aton, inet_addr, inet_make_addr library functions, or
/// directly with name resolver (see gethostbyname)
///
/// There are several special addresses:
/// INADDR_LOOPBACK (127.0.0.1)
//------------------------------------------------------------------------------
struct InternetSocketAddress : public ::sockaddr_in
{
  InternetSocketAddress(
    const uint16_t sin_port = 0,
    const uint16_t sin_family = AF_INET,
    const uint32_t sin_addr = INADDR_ANY);

  InternetSocketAddress();

  // Consider using &internet_socket_address for
  // InternetSocketAddress internet_socket_address, instead.
  const ::sockaddr_in* to_sockaddr_in() const
  {
    return reinterpret_cast<const ::sockaddr_in*>(this);
  }

  ::sockaddr_in* to_sockaddr_in()
  {
    return reinterpret_cast<::sockaddr_in*>(this);
  }

  const ::sockaddr* to_sockaddr() const
  {
    return reinterpret_cast<const ::sockaddr*>(this);
  }

  ::sockaddr* to_sockaddr()
  {
    return reinterpret_cast<::sockaddr*>(this);
  }

  friend std::ostream& operator<<(
    std::ostream& os,
    const InternetSocketAddress& internet_socket_address);
};

//------------------------------------------------------------------------------
/// \class InternetAddress
/// \brief ::sockaddr_in using encapsulation (composition)
//------------------------------------------------------------------------------
class InternetAddress
{
  public:

  InternetAddress(
    const uint16_t sin_family = AF_INET,
    const uint16_t sin_port = 0,
    const uint32_t sin_addr = INADDR_ANY);

  explicit InternetAddress(const uint16_t sin_port);

  InternetAddress();

  //----------------------------------------------------------------------------
  /// \brief user-defined conversion
  /// \details Make a reference when user wants to pass this user-defined
  /// converted InternetAddress into a function that takes a pointer, e.g.
  ///
  /// &(::sockaddr_in{some_internet_address})
  ///
  /// \ref https://en.cppreference.com/w/cpp/language/cast_operator
  //----------------------------------------------------------------------------
  operator ::sockaddr_in() const
  {
    return socket_address_internet_;
  }

  operator ::sockaddr*()
  {
    return reinterpret_cast<::sockaddr*>(&socket_address_internet_);
  }

  /// Accessors for Linux system call.

  const ::sockaddr_in* to_socket_address_internet_pointer() const
  {
    return &socket_address_internet_;
  }

  ::sockaddr_in* to_socket_address_internet_pointer()
  {
    return &socket_address_internet_;
  }

  const ::sockaddr_in* as_socket_address_internet_pointer() const
  {
    return reinterpret_cast<const ::sockaddr_in*>(this);
  }

  ::sockaddr_in* as_socket_address_internet_pointer()
  {
    return reinterpret_cast<::sockaddr_in*>(this);
  }

  const ::sockaddr* as_socket_address_pointer() const
  {
    return reinterpret_cast<const ::sockaddr*>(this);
  }

  ::sockaddr* as_socket_address_pointer()
  {
    return reinterpret_cast<::sockaddr*>(this);
  }

  friend std::ostream& operator<<(
    std::ostream& os,
    const InternetAddress& internet_address);

  private:

    ::sockaddr_in socket_address_internet_;
};

} // namespace Sockets

} // namespace IPC

#endif // _IPC_SOCKETS_INTERNET_ADDRESS_H_
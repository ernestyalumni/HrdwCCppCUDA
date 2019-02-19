//------------------------------------------------------------------------------
/// \file UnixDomainAddress.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Wrappers and classes for UNIX domain addresses.
/// \ref http://man7.org/linux/man-pages/man7/unix.7.html
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
#ifndef _IPC_SOCKETS_UNIX_DOMAIN_ADDRESS_H_
#define _IPC_SOCKETS_UNIX_DOMAIN_ADDRESS_H_

#include <cstddef> // std::size_t
#include <netinet/un.h>
#include <ostream>

namespace IPC
{

namespace Sockets
{

//------------------------------------------------------------------------------
/// \brief Derived class for ::sockaddr_un
/// \url http://man7.org/linux/man-pages/man7/unix.7.html
/// \details A UNIX domain socket address is represented in the following
/// structure:
/// 
/// struct sockaddr_un
/// {
///   sa_family_t sun_family; // address familly: AF_UNIX
///   char sun_path[108]; // pathname
/// };
///
/// sun_family field always contains AF_UNIX.
//------------------------------------------------------------------------------
struct UnixDomainSocketAddress : public ::sockaddr_un
{
  UnixDomainSocketAddress(
    const uint16_t sun_family,
    const std::string& sun_path);

  explicit UnixDomainSocketAddress(const std::string& sun_path);

  UnixDomainSocketAddress();

  // Consider using &unix_domain_socket_address for
  // UnixDomainSocketAddress unix_domain_socket_address, instead.
  const ::sockaddr_un* to_sockaddr_un() const
  {
    return reinterpret_cast<const ::sockaddr_un*>(this);
  }

  ::sockaddr_un* to_sockaddr_un()
  {
    return reinterpret_cast<::sockaddr_un*>(this);
  }

  const ::sockaddr* to_sockaddr() const
  {
    return reinterpret_cast<const ::sockaddr*>(this);
  }

  ::sockaddr* to_sockaddr()
  {
    return reinterpret_cast<::sockaddr*>(this);
  }

  std::size_t address_length() const
  {
    return std::strlen(this->sun_path) + sizeof(this->sun_family);
  }

  friend std::ostream& operator<<(
    std::ostream& os,
    const UnixDomainSocketAddress& unix_domain_socket_address);
};

//------------------------------------------------------------------------------
/// \class UnixDomainAddress
/// \brief ::sockaddr_un using encapsulation (composition)
//------------------------------------------------------------------------------
class UnixDomainAddress
{
  public:

  UnixDomainAddress(
    const uint16_t sun_family,
    const std::string& sun_path);

  explicit UnixDomainAddress(const std::string& sun_path);

  UnixDomainAddress();

  //----------------------------------------------------------------------------
  /// \brief user-defined conversion
  /// \details Make a reference when user wants to pass this user-defined
  /// converted InternetAddress into a function that takes a pointer, e.g.
  ///
  /// &(::sockaddr_un{some_unix_address})
  ///
  /// \ref https://en.cppreference.com/w/cpp/language/cast_operator
  //----------------------------------------------------------------------------
  operator ::sockaddr_un() const
  {
    return socket_address_unix_;
  }

  operator ::sockaddr*()
  {
    return reinterpret_cast<::sockaddr*>(&socket_address_internet_);
  }

  /// Accessors for Linux system call.

  const ::sockaddr_un* to_socket_address_unix_pointer() const
  {
    return &socket_address_internet_;
  }

  ::sockaddr_un* to_socket_address_unix_pointer()
  {
    return &socket_address_internet_;
  }

  const ::sockaddr_un* as_socket_address_unix_pointer() const
  {
    return reinterpret_cast<const ::sockaddr_un*>(this);
  }

  ::sockaddr_un* as_socket_address_unix_pointer()
  {
    return reinterpret_cast<::sockaddr_un*>(this);
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
    const UnixDomainAddress& unix_domain_address);

  private:

    ::sockaddr_un socket_address_unix_;
};

} // namespace Sockets

} // namespace IPC

#endif // _IPC_SOCKETS_UNIX_DOMAIN_ADDRESS_H_
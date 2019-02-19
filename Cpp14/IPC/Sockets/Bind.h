//------------------------------------------------------------------------------
/// \file Bind.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Wrapper for ::bind as a C++ functor.
/// \ref http://man7.org/linux/man-pages/man2/bind.2.html
/// \details Bind a name to a socket.
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
///   g++ -I ../../ -std=c++14 Socket.cpp Bind_main.cpp \
///     ../../Utilities/ErrorHandling.cpp ../../Utilities/Errno.cpp -o Bind_main
//------------------------------------------------------------------------------
#ifndef _IPC_SOCKETS_BIND_H_
#define _IPC_SOCKETS_BIND_H_

#include "InternetAddress.h"
#include "Socket.h"
#include "Utilities/ErrorHandling.h" // HandleReturnValue

#include <cstring> // std::memset 
#include <type_traits>

namespace IPC
{

namespace Sockets
{

//------------------------------------------------------------------------------
/// \class Bind
/// \url http://man7.org/linux/man-pages/man2/bind.2.html
/// \details
/// #include <sys/socket.h>
/// int bind(int sockfd, const struct sockaddr *addr, socklen_t addrlen)
///
/// When a socket is created with ::socket, it exists in name space (address
/// family), but has no address assigned to it.
/// bind() assigns address specified by addr to socket referred to by fd sockfd.
/// addrlen specified size, in bytes, of address structure pointed to by addr.
///
/// It's normally necessary to assign a local address using ::bind() before a
/// SOCK_STREAM socket may receive connections (see ::accept).
///
/// Actual structure passed for addr argument will depend on address family.
//------------------------------------------------------------------------------
template <typename Implementation>
class Bind
{
  public:

    using Socket = IPC::Sockets::Socket;

    void operator()(Socket& socket)
    {
      static_cast<Implementation&>(*this).operator()(socket);
    }

  protected:

    //--------------------------------------------------------------------------
    /// \class HandleBind
    /// \ref http://man7.org/linux/man-pages/man2/bind.2.html
    /// \brief On success, 0 returned. On error, -1 returned, and errno is set
    /// appropriately.
    /// \details ERRORS:
    /// EACCES Address is protected, user isn't superuser.
    /// EADDRINUSE Given address already in use.
    /// EADDRINUSE (Internet domain sockets) The port number was specified as 0
    /// in socket address structure, but, upon attempting to bind to an
    /// ephemeral port, it was determined that all port numbers in ephemeral
    /// port range are currently in use.
    /// EBADF sockfd isn't a valid fd.
    /// EINVAL Socket already bound to an address.
    /// EINVAL addrlen is wrong, or addr isn't a valid address for this socket's
    /// domain.
    //--------------------------------------------------------------------------
    class HandleBind : public Utilities::ErrorHandling::HandleReturnValue
    {
      public:

        HandleBind() = default;

        void operator()(const int result)
        {
          this->operator()(
            result,
            "Assign name (address) to a socket (::bind)");
        }

      private:

        using HandleReturnValue::operator();
    };
}; 

template <typename AddressFamily>
class BindAddressFamily : public Bind<BindAddressFamily<AddressFamily>>
{
  protected:

    using HandleBind =
      typename Bind<BindAddressFamily<AddressFamily>>::HandleBind;

  public:

    BindAddressFamily()
    {
      if (std::is_trivially_copyable<AddressFamily>::value)
      {
        // Clear structure
        std::memset(&address_, 0, sizeof(AddressFamily));
      }
    }

    //--------------------------------------------------------------------------
    /// \fn BindAddressFamily(const AddressFamily& address)
    /// \brief Constructor
    /// \ref https://en.cppreference.com/w/cpp/string/byte/memset
    //--------------------------------------------------------------------------    
    BindAddressFamily(const AddressFamily& address):
      address_{address}
    {}

    void operator()(Socket& socket)
    {
      const int sfd {socket.fd()};

      const int result {::bind(
        sfd,
        reinterpret_cast<::sockaddr*>(&address_),
        sizeof(AddressFamily))};

      HandleBind()(result);
    }

    void operator()(Socket& socket, socklen_t address_length)
    {
      const int sfd {socket.fd()};

      const int result {::bind(
        sfd,
        reinterpret_cast<::sockaddr*>(&address_),
        address_length)};

      HandleBind()(result);
    }

  private:

    AddressFamily address_;
}; // class BindAddressFamily

#if 0

template <class InternetAddressImplementation>
class BindToInternetAddress
{
  public:

    BindToInternetAddress() = default;

  private:

    InternetAddressImplementation internet_address_;
};
#endif 

} // namespace Sockets

} // namespace IPC

#endif // _IPC_SOCKETS_BIND_H_
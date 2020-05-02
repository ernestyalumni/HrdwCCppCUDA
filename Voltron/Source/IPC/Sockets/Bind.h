//------------------------------------------------------------------------------
/// \file Bind.h
/// \author Ernest Yeung
/// \brief Bind, or assign the address to the socket referred to by a file
/// descriptor. Traditionally called "assigning a name to a socket."
///-----------------------------------------------------------------------------
#ifndef IPC_SOCKETS_BIND_H
#define IPC_SOCKETS_BIND_H

#include "IPC/Sockets/InternetAddress.h"
#include "IPC/Sockets/Socket.h"
#include "Utilities/ErrorHandling/ErrorNumber.h"

#include <optional>

namespace IPC
{
namespace Sockets
{

//------------------------------------------------------------------------------
/// \class Bind
//------------------------------------------------------------------------------

class Bind
{
  public:

    Bind(const InternetSocketAddress& internet_socket_address):
      internet_socket_address_{internet_socket_address}
    {}

    std::optional<Utilities::ErrorHandling::ErrorNumber> operator()(
      Socket& socket);

    InternetSocketAddress internet_socket_address() const
    {
      return internet_socket_address_;
    }

  private:

    //--------------------------------------------------------------------------
    /// \brief Return value of ::bind: on success, 0 is returned. On error, -1
    /// is returned and errno is set appropriately.
    /// \details 
    /// EACCES Address is protected, and user is not superuser.
    /// EADDRINUSE Given address is already in use.
    /// EADDRINUSE (Internet domain sockets) The port number was specified as 0
    /// in socket address structure, but upon attempting to bind to an
    /// ephemeral port, it was determined that all port numbers in ephemeral
    /// port range are currently in use.
    /// EBADF sockfd isn't a valid fd.
    /// EINVAL Socket already bound to address.
    /// EINVAL addrlen is wrong, or addr isn't a valid address for this socket's
    /// domain.
    /// ENOTSOCK fd sockfd doesn't refer to a socket.
    //--------------------------------------------------------------------------
    class HandleBind
    {
      public:

        HandleBind();

        std::optional<Utilities::ErrorHandling::ErrorNumber> operator()(
          const int return_value);

      private:

        Utilities::ErrorHandling::ErrorNumber error_number_;
    };

    InternetSocketAddress internet_socket_address_;
};

} // namespace Sockets
} // namespace IPC

#endif // IPC_SOCKETS_BIND_H
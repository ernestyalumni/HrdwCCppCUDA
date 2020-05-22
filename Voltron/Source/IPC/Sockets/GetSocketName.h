//------------------------------------------------------------------------------
/// \file GetSocketName.h
/// \author Ernest Yeung
/// \brief ::getsockname wrapper.
/// \ref http://man7.org/linux/man-pages/man2/getsockname.2.html
//------------------------------------------------------------------------------
#ifndef IPC_SOCKETS_GET_SOCKET_NAME_H
#define IPC_SOCKETS_GET_SOCKET_NAME_H

#include "IPC/Sockets/InternetAddress.h"
#include "IPC/Sockets/Socket.h"
#include "Utilities/ErrorHandling/ErrorNumber.h"

#include <optional>
#include <tuple>
#include <utility>

namespace IPC
{
namespace Sockets
{

class GetSocketName
{
  public:

    GetSocketName();

    std::optional<Utilities::ErrorHandling::ErrorNumber> operator()(
      const Socket& socket);

    /*
    std::pair<
      std::optional<Utilities::ErrorHandling::ErrorNumber>,
      std::optional<std::tuple<InternetSocketAddress, socklen_t>>
      > operator()(const Socket& socket);
    */

    std::optional<std::tuple<InternetSocketAddress, socklen_t>>
      socket_address() const
    {
      return socket_address_;
    }

  private:

    //--------------------------------------------------------------------------
    /// \brief Return 0 on on success. On error, -1 is returned and errno is set
    /// appropriately.
    /// \details 
    /// EBADF Argument sockfd isn't a valid fd.
    /// EFAULT addr argument points to memory not in valid part of process
    /// address space.
    /// EINVAL addrlen is invalid (e.g. is negative)
    /// ENOBUFS Insufficient resources were available in system to perform
    /// operation.
    /// ENOTSOCK fd sockfd doesn't refer to socket.
    //--------------------------------------------------------------------------
    class HandleGetSocketName
    {
      public:

        HandleGetSocketName();

        std::optional<Utilities::ErrorHandling::ErrorNumber> operator()(
          const int return_value);

      private:

        Utilities::ErrorHandling::ErrorNumber error_number_;
    };

    std::optional<std::tuple<InternetSocketAddress, socklen_t>> socket_address_;
};

} // namespace Sockets
} // namespace IPC

#endif // IPC_SOCKETS_GET_SOCKET_NAME_H
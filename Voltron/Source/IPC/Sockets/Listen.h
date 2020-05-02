//------------------------------------------------------------------------------
/// \file Listen.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref http://man7.org/linux/man-pages/man2/listen.2.html
//------------------------------------------------------------------------------
#ifndef IPC_SOCKETS_LISTEN_H
#define IPC_SOCKETS_LISTEN_H

#include "IPC/Sockets/Socket.h"
#include "Utilities/ErrorHandling/ErrorNumber.h"

#include <optional>

namespace IPC
{
namespace Sockets
{

class MakeListen
{
  public:

    //--------------------------------------------------------------------------
    /// \param backlog_length Defines the maximum length to which the queue of
    /// pending connections for sockfd of Socket may grow.
    //--------------------------------------------------------------------------
    explicit MakeListen(const int backlog_length);

    std::optional<Utilities::ErrorHandling::ErrorNumber> operator()(
      const Socket& socket);

    int backlog_length() const
    {
      return backlog_length_;
    }

  private:

    //--------------------------------------------------------------------------
    /// \brief Return 0 on on success. On error, -1 is returned and errno is set
    /// appropriately.
    /// \details 
    /// EADDRINUSE Another socket is already listening on same port
    /// EADDRINUSE (Internet domain sockets) The socket referred to by sockfd
    /// had not previously been bound to address and, upon attempting to bind it
    /// to an ephemeral port, it was determined that all port numbers in
    /// ephemeral port range are currently in use.
    /// EBADF Argument sockfd isn't a valid fd.
    /// ENOTSOCK fd sockfd doesn't refer to socket.
    /// EOPNOTSUPP Socket isn't type that supports ::listen() operation.
    //--------------------------------------------------------------------------
    class HandleListen
    {
      public:

        HandleListen();

        std::optional<Utilities::ErrorHandling::ErrorNumber> operator()(
          const int return_value);

      private:

        Utilities::ErrorHandling::ErrorNumber error_number_;
    };

    int backlog_length_;
};

} // namespace Sockets
} // namespace IPC

#endif // IPC_SOCKETS_LISTEN_H
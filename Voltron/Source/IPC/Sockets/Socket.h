//------------------------------------------------------------------------------
/// \file Socket.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref https://linux.die.net/man/2/socket
//------------------------------------------------------------------------------
#ifndef IPC_SOCKETS_SOCKET_H
#define IPC_SOCKETS_SOCKET_H

#include "IPC/Sockets/ParameterFamilies.h"
#include "Utilities/ErrorHandling/ErrorHandling.h"

#include <ostream>

namespace IPC
{

namespace Sockets
{

//------------------------------------------------------------------------------
/// \brief Creates an endpoint for communication
//------------------------------------------------------------------------------
class Socket
{
  public:

    //--------------------------------------------------------------------------
    /// \fn Socket
    /// \brief Constructor matching ::socket function signature.
    /// \url http://man7.org/linux/man-pages/man2/socket.2.html
    /// \details The protocol specifies a particular protocol to be used with
    /// the socket. Normally, only a single protocol exists to support a
    /// particular socket type within a given protocol family, in which case
    /// protocol can be specified as 0.
    //--------------------------------------------------------------------------
    Socket(const int domain, const int type, const int protocol = 0);

    Socket(const Domains domain, const Types, const int protocol = 0);

    virtual ~Socket();

    // Accessors

    int domain() const
    {
      return domain_;
    }

    int type() const
    {
      return type_;
    }

    int protocol() const
    {
      return protocol_;
    }

    friend std::ostream& operator<<(std::ostream& os, const Socket& socket);

    const int fd() const
    {
      return fd_;
    }

  protected:

    //--------------------------------------------------------------------------
    /// \brief Return value of ::socket: on success, a fd for the new socket is
    /// returned. On error, -1 is returned and errno is set appropriately.
    /// \details 
    /// EACCES Permission to create a socket of the specified type and/or
    /// protocol is defined.
    /// EAFNOSUPPORT The implementation does not support the specified address
    /// family.
    /// EINVAL Unknown protocol, or protocol family not available
    /// EINVAL INvalid flags in type
    /// EMFILE The per-process limit on the number of open fds has been reached.
    /// ENFILE The system-wide limit on total number of open files has been
    /// reached.
    /// ENOBUFS or ENOMEM Insufficient memory is available. The socket cannot be
    /// created until sufficient resources are freed.
    /// EPROTONOSUPPORT The protocol type or specified protocol isn't supported
    /// within this domain.
    //--------------------------------------------------------------------------
    class HandleSocket : public Utilities::ErrorHandling::HandleReturnValue
    {
      public:

        HandleSocket();

        void operator()(const int result);

      private:

        using HandleReturnValue::operator();
    };

    int create_socket(const int domain, const int type, const int protocol);

  private:

    int domain_;
    int type_;
    int protocol_;

    int fd_;
};


} // namespace Sockets
} // namespace IPC

#endif // IPC_SOCKETS_SOCKET_H

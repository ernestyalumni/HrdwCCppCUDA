//------------------------------------------------------------------------------
/// \file Send.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief Send a message on a socket.
/// \ref https://linux.die.net/man/2/sendto
//------------------------------------------------------------------------------
#ifndef IPC_SOCKETS_SEND_H
#define IPC_SOCKETS_SEND_H

#include "IPC/Sockets/InternetAddress.h"
#include "IPC/Sockets/Socket.h"
#include "Utilities/ErrorHandling/ErrorNumber.h"

#include <array>
#include <optional>
#include <sys/socket.h> // socklen_t
#include <tuple>
#include <utility>

namespace IPC
{
namespace Sockets
{

//------------------------------------------------------------------------------
/// \class SendTo
/// \brief Wrapper for ::sendto
/// \ref https://linux.die.net/man/2/sendto
/// https://linux.die.net/man/3/sendto
/// \details ssize_t sendto(int sockfd, const void* buf, size_t len, int flags,
/// struct sockaddr *src_addr, socklen_t addrlen);
//------------------------------------------------------------------------------
// N is the size of the buffer to "read" into. Arbitrary default buffer size of
// 2048 is from
// https://www.cs.rutgers.edu/~pxk/417/notes/sockets/demo-udp-04.html
template <std::size_t N = 2048>
class SendTo
{
  public:

    SendTo(const int flags = 0);

    std::pair<
      std::optional<Utilities::ErrorHandling::ErrorNumber>,
      std::optional<std::tuple<std::size_t, socklen_t>>
      > operator()(const InternetSocketAddress& destination_address);

    // Accessors and Setters.

    InternetSocketAddress destination_address() const
    {
      return destination_address_;
    }

    void destination_address(const InternetSocketAddress& address)
    {
      destination_address_ = address;
    }

    //--------------------------------------------------------------------------
    /// \ref https://en.cppreference.com/w/cpp/language/nested_types
    /// \details Name lookup from member function of nested class visits scope
    /// of enclosing class after examining scope of nested class. Like any
    /// member of enclosing class, nested class has access to all names
    /// (private, protected, etc.) to which enclosing class has access, but
    /// otherwise independent and has no special access to this pointer of
    /// enclosing class.
    //--------------------------------------------------------------------------
    class SendMessage
    {
      public:

        SendMessage();

        operator()(const std::array<char, N>& buffer);

        // Accessors and Setters.

        std::array<char, N> buffer() const
        {
          return buffer_;
        }

        void buffer(const std::array<char, N>& buffer)
        {
          buffer_ = buffer;
        }

      private:

        //----------------------------------------------------------------------
        /// \ref https://linux.die.net/man/3/sendto
        /// \brief Upon success, sendto() return number of bytes sent.
        /// Otherwise, return -1, and errno set to indicate error.
        /// \details 
        /// EAFNOSUPPORT - addresses in specified address family can't be used
        /// with this socket.
        /// EAGAIN or EWOULDBLOCK - socket's fd marked O_NONBLOCK and requested
        /// operation would block.
        /// EBADF - socket argument not a valid fd
        /// ECONNRESET - connection forcibly closed by peer
        /// EINTR - Signal interrupted sendto() before any data transmitted.
        /// EMSGSIZE - message too large to be sent all at once, as socket
        /// requires.
        /// ENOTSOCK - socket argument doesn't refer to socket.
        /// EOPNOTSUPP - specified flags aren't supported for this socket type.
        /// EPIPE - socket is shut down for writing, or socket is
        /// connection-mode and is no longer connected. In latter case, and if
        /// socket of type SOCK_STREAM, SIGPIPE signal generated to calling
        /// thread.
        /// EIO - IO error while reading from or writing to file system
        /// ELOOP - loop exists in symbolic links encountered during resolution
        /// of pathname in socket address.
        /// ENAMETOOLONG
        /// ENOENT
        /// ENOTDIR
        /// ENOBUFS - Insufficient resources available in system to perform
        /// operation.
        /// ENOMEM - insufficient memory available to fulfill request.
        //----------------------------------------------------------------------
        class HandleSendMessage
        {
          public:

            HandleSendMessage();

            std::optional<Utilities::ErrorHandling::ErrorNumber> operator()(
              const std::size_t return_value);

          private:

            Utilities::ErrorHandling::ErrorNumber error_number_;
        };
    };

  private:

    // Address for the destination.
    InternetSocketAddress destination_address_;

    int flags_;
};

} // namespace Sockets
} // namespace IPC

#endif // IPC_SOCKETS_RECEIVE_H
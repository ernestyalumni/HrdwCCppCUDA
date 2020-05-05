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

    class SendMessage;

    SendTo(const int flags = 0):
      flags_{flags}
    {}

    SendMessage operator()(const InternetSocketAddress& address)
    {
      destination_address(address);

      return SendMessage{flags_, address};
    }

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

        using ErrorNumber = Utilities::ErrorHandling::ErrorNumber;

        SendMessage(const int flags, const InternetSocketAddress& address):
          destination_address_{address},
          flags_{flags}
        {}

        auto operator()(const std::array<char, N>& buffer)
        {
          const int flags {flags_};
          const InternetSocketAddress& destination_address {
            destination_address_};

          auto send_on_socket =
            [flags, &destination_address, &buffer](Socket& socket) ->
              std::pair<
                std::optional<ErrorNumber>,
                std::optional<ssize_t>>
            {
              const ssize_t return_value {
                ::sendto(
                  socket.fd(),
                  buffer.data(),
                  buffer.size(),
                  flags,
                  destination_address.to_sockaddr(),
                  destination_address.address_size())};

              std::optional<ErrorNumber> error_number {
                HandleSendMessage()(return_value)};

              if (error_number)
              {
                return std::make_pair<
                  std::optional<ErrorNumber>,
                  std::optional<ssize_t>
                  >(std::move(error_number), std::nullopt);
              }
              else
              {
                return std::make_pair<
                  std::optional<ErrorNumber>,
                  std::optional<ssize_t>
                  >(
                    std::nullopt,
                    std::make_optional<ssize_t>(return_value));
              }
            };
          return send_on_socket;
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

            HandleSendMessage() = default;

            std::optional<ErrorNumber> operator()(
              const std::size_t return_value)
            {
              if (return_value < 0)
              {
                const auto error_number = ErrorNumber{};

                return std::make_optional<ErrorNumber>(error_number);
              }
              else
              {
                return std::nullopt;
              }                
            }

          private:

            ErrorNumber error_number_;
        };

      InternetSocketAddress destination_address_;
      int flags_;
    };

  private:

    // Address for the destination.
    InternetSocketAddress destination_address_;

    int flags_;
};

} // namespace Sockets
} // namespace IPC

#endif // IPC_SOCKETS_RECEIVE_H
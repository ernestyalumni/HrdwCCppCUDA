//------------------------------------------------------------------------------
/// \file Receive.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief Receive message from a socket.
/// \ref https://linux.die.net/man/2/recvfrom
//------------------------------------------------------------------------------
#ifndef IPC_SOCKETS_RECEIVE_H
#define IPC_SOCKETS_RECEIVE_H

#include "IPC/Sockets/InternetAddress.h"
#include "IPC/Sockets/Socket.h"
#include "Utilities/ErrorHandling/ErrorNumber.h"

#include <array>
#include <optional>
#include <sys/socket.h> // socklen_t
#include <tuple>
#include <utility> // std::pair

namespace IPC
{
namespace Sockets
{

//------------------------------------------------------------------------------
/// \class ReceiveFrom
/// \brief Wrapper for ::recvfrom
/// \ref https://linux.die.net/man/2/recvfrom
/// https://linux.die.net/man/3/recvfrom
/// \details ssize_t recvfrom(int sockfd, void* buf, size_t len, int flags,
/// struct sockaddr *src_addr, socklen_t *addrlen);
/// buffer points to the buffer where message should be stored.
/// length specifies length in bytes of buffer pointed to by buffer argument
/// flags specifies type of message reception. Values of this argument formed by
/// logically OR'ing 0 or more of following values:
/// MSG_PEEK peeks at incoming message. data treated as unread and next
/// recvfrom() or similar function shall still return this data
/// MSG_OOB Requests out-of-band data. Signifiance and semantics of out-of-band
/// data are protocol-specific.
/// MSG_WAITALL On SOCK_STREAM sockets this requests that function block until
/// full amount of data can be returned. Function may return smaller amount of
/// data if socket is a message-based socket, if signal is caught, if connection
/// is terminated, if MSG_PEEK was specified, or if error pending for socket.
//------------------------------------------------------------------------------
// N is the size of the buffer to "read" into. Arbitrary default buffer size of
// 2048 is from
// https://www.cs.rutgers.edu/~pxk/417/notes/sockets/demo-udp-04.html
template <std::size_t N = 2048>
class ReceiveFrom
{
  public:

    ReceiveFrom(const int flags = 0);

    std::pair<
      std::optional<Utilities::ErrorHandling::ErrorNumber>,
      std::optional<std::tuple<std::size_t, socklen_t>>
      > operator()(Socket& socket);

    // Accessors

    std::array<char, N> buffer() const
    {
      return buffer_;
    }

    InternetSocketAddress sender_address() const
    {
      return sender_address_;
    }

  private:

    //--------------------------------------------------------------------------
    /// \ref https://linux.die.net/man/3/recvfrom
    /// \brief Return length of message in bytes. If no messages available to be
    /// received, and peer has performed orderly shutdown, return 0. Otherwise,
    /// return -1 and set errno to indicate error.
    /// \details 
    /// EAGAIN or EWOULDBLOCK - socket's fd marked O_NONBLOCK and no data
    /// waiting to be received; or MSG_OOB set and no out-of-band data available
    /// and either socket's fd marked O_NONBLOCK or socket doesn't support
    /// blocking to await out-of-band data.
    /// EBADF - socket argument not a valid fd
    /// ECONNRESET - connection forcibly closed by peer
    /// EINTR - MSG_OOB flag set and no out-of-band data available.
    /// ENOTCONN - receive attempted on connection-mode socket that's not
    /// connected.
    /// ENOTSOCK - socket argument doesn't refer to socket.
    /// EOPNOTSUPP - specified flags aren't supported for this socket type.
    /// ETIMEDOUT - connection timed out during connection establishment, or due
    /// to transmission timeout on active connection.
    /// EIO - IO error while reading from or writing to file system
    /// ENOBUFS - Insufficient resources available in system to perform
    /// operation.
    /// ENOMEM - insufficient memory available to fulfill request.
    //--------------------------------------------------------------------------
    class HandleReceiveFrom
    {
      public:

        HandleReceiveFrom();

        std::optional<Utilities::ErrorHandling::ErrorNumber> operator()(
          const std::size_t return_value);

      private:

        Utilities::ErrorHandling::ErrorNumber error_number_;
    };

    std::array<char, N> buffer_;
    // Address of the sender.
    InternetSocketAddress sender_address_;
    // Also gives the "state" of ::recvfrom
    std::optional<Utilities::ErrorHandling::ErrorNumber> error_number_;

    int flags_;
};

class ReceivingOn
{
  public:

    template <std::size_t N>
    class ReceivedFrom
    {
      public:

        using ErrorNumber = Utilities::ErrorHandling::ErrorNumber;

        struct Receipt
        {
          std::array<char, N> buffer_;
          InternetSocketAddress sender_address_;
          ssize_t received_bytes_;
          socklen_t size_of_sender_address_;
        };

        explicit ReceivedFrom(Socket& socket, const int flags):
          flags_{flags},
          socket_fd_{socket.fd()}
        {}

        std::pair<std::optional<ErrorNumber>, std::optional<Receipt>> operator()()
        {
          std::array<char, N> buffer;
          InternetSocketAddress sender_address;
          socklen_t size_of_sender_address {sender_address.address_size()};

          // ssize_t aka long int.
          ssize_t return_value {
            ::recvfrom(
              socket_fd_,
              buffer.data(),
              buffer.size(),
              flags_,
              sender_address.to_sockaddr(),
              &size_of_sender_address)};

          std::optional<ErrorNumber> error_number {HandleReceivedFrom()(return_value)};

          if (error_number)
          {
            return std::make_pair<
              std::optional<ErrorNumber>,
              std::optional<Receipt>
              >(std::move(error_number), std::nullopt);
          }
          else
          {
            Receipt received_details {
              buffer,
              sender_address,
              return_value,
              size_of_sender_address};

            return std::make_pair<
              std::optional<ErrorNumber>,
              std::optional<Receipt>
              >(std::move(error_number), std::move(received_details));
          }
        }

      private:

        class HandleReceivedFrom
        {
          public:

            HandleReceivedFrom() = default;

            std::optional<ErrorNumber> operator()(const std::size_t return_value)
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

        int flags_;
        int socket_fd_;
    };

    ReceivingOn(const int flags = 0);

    template <std::size_t N = 2048>
    ReceivedFrom<N> operator()(Socket& socket)
    {
      return ReceivedFrom<N>{socket, flags_};
    }

  private:

    int flags_;
};

} // namespace Sockets
} // namespace IPC

#endif // IPC_SOCKETS_RECEIVE_H
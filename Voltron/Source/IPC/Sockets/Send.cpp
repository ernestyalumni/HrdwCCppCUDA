//------------------------------------------------------------------------------
/// \file Send.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref https://linux.die.net/man/2/sendto
//------------------------------------------------------------------------------
#include "Send.h"

#include "IPC/Sockets/InternetAddress.h"
#include "IPC/Sockets/Socket.h"
#include "Utilities/ErrorHandling/ErrorNumber.h"

#include <array>
#include <optional>
#include <sys/socket.h> // ::getsockname
#include <utility> // std::pair

using Utilities::ErrorHandling::ErrorNumber;

namespace IPC
{
namespace Sockets
{

template <std::size_t N>
SendTo<N>::SendTo(const int flags = 0):
  flags_{flags}
{}

template <std::size_t N>
SendTo<N>::operator()(const InternetSocketAddress& destination_address)
{
  destination_address(destination_address);

}

template <std::size_t N>
SendTo<N>::SendMessage::SendMessage():
  buffer_{}
{}

template <std::size_t N>
SendTo<N>::SendMessage::operator()(const std::array<char, N>& buffer)
{
  auto send_on_socket =
    [flags_, &destination_address_, &buffer](Socket& socket) ->
      std::pair<
        std::optional<ErrorNumber>,
        std::optional<std::size_t>>
    {
      const std::size_t return_value {
        ::sendto(socket.fd(), buffer.data(), buffer.size(),
          flags_, destination_address_.to_sockaddr(),
          destination_address_.address_size())};

      std::optional<ErrorNumber> error_number {HandleSendTo()(return_value)};

      if (error_number)
      {
        return std::make_pair<
          std::optional<ErrorNumber>,
          std::optional<std::size_t>
          >(std::move(error_number), std::nullopt);
      }
      else
      {
        return std::make_pair<
          std::optional<ErrorNumber>,
          std::optional<std::size_t>
          >(
            std::move(error_number),
            std::make_optional<std::size_t>(return_value));
      }
    };
  return send_on_socket;
}


std::optional<ErrorNumber> MakeListen::operator()(const Socket& socket)
{
  const int return_value {::listen(socket.fd(), backlog_length_)};

  return HandleListen()(return_value);
}

template <std::size_t N>
SendTo::SendMessage<N>::HandleSendMessage::HandleSendMessage() = default;

template <std::size_t N>
std::optional<ErrorNumber>
  SendTo::SendMessage<N>::HandleSendMessage::operator()(
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

} // namespace Sockets
} // namespace IPC

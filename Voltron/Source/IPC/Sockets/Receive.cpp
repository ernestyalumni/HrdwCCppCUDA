//------------------------------------------------------------------------------
/// \file Receive.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref https://linux.die.net/man/2/recvfrom
//------------------------------------------------------------------------------
#include "Receive.h"

#include "IPC/Sockets/Socket.h"
#include "Utilities/ErrorHandling/ErrorNumber.h"

#include <optional>
#include <sys/socket.h> // ::recvfrom
#include <tuple>
#include <utility>

using Utilities::ErrorHandling::ErrorNumber;

namespace IPC
{
namespace Sockets
{

template <std::size_t N>
ReceiveFrom<N>::ReceiveFrom(const int flags):
  buffer_{},
  sender_address_{},
  error_number_{std::nullopt},
  flags_{flags}
{}

// cf. https://en.cppreference.com/w/cpp/container/array/data
// std::array<T, N>::data
// Returns pointer to underlying array serving as element storage.
template <std::size_t N>
std::pair<
  std::optional<ErrorNumber>,
  std::optional<std::tuple<std::size_t, socklen_t>>
  > ReceiveFrom<N>::operator()(Socket& socket)
{
  socklen_t size_of_sender_address {sender_address_.address_size()};

  std::size_t return_value {
    ::recvfrom(
      socket.fd(),
      buffer_.data(),
      buffer_.size(),
      flags_, 
      sender_address_.to_sockaddr(),
      &size_of_sender_address)};

  std::optional<ErrorNumber> error_number {HandleReceiveFrom()(return_value)};

  if (error_number)
  {
    return std::make_pair<
      std::optional<ErrorNumber>,
      std::optional<std::tuple<std::size_t, socklen_t>>
      >(std::move(error_number), std::nullopt);
  }
  else
  {
    return std::make_pair<
      std::optional<ErrorNumber>,
      std::optional<std::tuple<std::size_t, socklen_t>>
      >(
        std::move(error_number),
        std::make_optional<std::tuple<std::size_t, socklen_t>>(
          std::make_tuple<std::size_t, socklen_t>(
            std::move(return_value),
            std::move(size_of_sender_address))));
  }
}

template <std::size_t N>
std::optional<ErrorNumber> ReceiveFrom<N>::HandleReceiveFrom::operator()(
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

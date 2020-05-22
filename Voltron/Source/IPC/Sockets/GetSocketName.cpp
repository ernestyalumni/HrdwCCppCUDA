//------------------------------------------------------------------------------
/// \file GetSocketName.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref http://man7.org/linux/man-pages/man2/getsockname.2.html
//------------------------------------------------------------------------------
#include "GetSocketName.h"

#include "IPC/Sockets/InternetAddress.h"
#include "IPC/Sockets/Socket.h"
#include "Utilities/ErrorHandling/ErrorNumber.h"

#include <optional>
#include <sys/socket.h> // ::getsockname
#include <tuple>
#include <utility>

using Utilities::ErrorHandling::ErrorNumber;

namespace IPC
{
namespace Sockets
{

GetSocketName::GetSocketName():
  socket_address_{std::nullopt}
{}

std::optional<ErrorNumber> GetSocketName::operator()(const Socket& socket)
{
  InternetSocketAddress internet_socket_address;

  // cf. http://man7.org/linux/man-pages/man2/getsockname.2.html
  // This, the "addrlen argument", needs to be initialized to indicate amount of
  // space (in bytes) pointed to by addr, which is InternetSocketAddress size.
  socklen_t size_of_address {sizeof(internet_socket_address)};

  const int return_value {
    ::getsockname(
      socket.fd(),
      internet_socket_address.to_sockaddr(),
      &size_of_address)};

  std::optional<ErrorNumber> error_number {
    HandleGetSocketName()(return_value)};

  if (error_number)
  {
    return std::forward<std::optional<ErrorNumber>>(error_number);
  }
  else
  {
    socket_address_ =
      std::make_optional<std::tuple<InternetSocketAddress, socklen_t>>(
        std::make_tuple<InternetSocketAddress, socklen_t>(
          std::forward<InternetSocketAddress>(internet_socket_address),
          std::move(size_of_address)));

    return std::nullopt;
  }
}

/*
std::pair<
  std::optional<ErrorNumber>,
  std::optional<std::tuple<InternetSocketAddress, socklen_t>>
  > GetSocketName::operator()(const Socket& socket)
{
  InternetSocketAddress internet_socket_address;

  socklen_t size_of_address;

  const int return_value {
    ::getsockname(
      socket.fd(),
      internet_socket_address.to_sockaddr(),
      &size_of_address)};

  std::optional<ErrorNumber> error_number {
    HandleGetSocketName()(return_value)};

  if (error_number)
  {
    return std::make_pair<
      std::optional<ErrorNumber>,
      std::optional<std::tuple<InternetSocketAddress, socklen_t>>
      >(std::forward<std::optional<ErrorNumber>>(error_number), std::nullopt);
  }
  else
  {
    return std::make_pair<
      std::optional<ErrorNumber>,
      std::optional<std::tuple<InternetSocketAddress, socklen_t>>
      >(std::forward<std::optional<ErrorNumber>>(error_number),
        std::make_optional<std::tuple<InternetSocketAddress, socklen_t>>(
          std::make_tuple<InternetSocketAddress, socklen_t>(
            std::forward<InternetSocketAddress>(internet_socket_address),
            std::move(size_of_address))));
  }
}
*/

GetSocketName::HandleGetSocketName::HandleGetSocketName() = default;

std::optional<ErrorNumber> GetSocketName::HandleGetSocketName::operator()(
  const int return_value)
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

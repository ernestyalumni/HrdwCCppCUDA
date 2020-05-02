//------------------------------------------------------------------------------
/// \file Bind.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref http://man7.org/linux/man-pages/man2/bind.2.html
//------------------------------------------------------------------------------
#include "Bind.h"

#include "IPC/Sockets/Socket.h"
#include "Utilities/ErrorHandling/ErrorNumber.h"

#include <optional>
#include <sys/socket.h> // ::bind

using Utilities::ErrorHandling::ErrorNumber;

namespace IPC
{
namespace Sockets
{

//------------------------------------------------------------------------------
/// \ref http://man7.org/linux/man-pages/man2/bind.2.html
/// \details int ::bind(int sockfd, const struct sockaddr *addr,
/// socklen_t addrlen)
//------------------------------------------------------------------------------
std::optional<ErrorNumber> Bind::operator()(Socket& socket)
{
  const int return_value {
    ::bind(
      socket.fd(),
      internet_socket_address_.to_sockaddr(),
      internet_socket_address_.address_size())};

  return HandleBind()(return_value);
}

Bind::HandleBind::HandleBind() = default;

//------------------------------------------------------------------------------
/// \ref http://man7.org/linux/man-pages/man2/bind.2.html
/// \details Return Value of ::bind:
/// On success, 0 is returned. On error, -1 is returned and errno is set
/// appropriately.
//------------------------------------------------------------------------------
std::optional<ErrorNumber> Bind::HandleBind::operator()(const int return_value)
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

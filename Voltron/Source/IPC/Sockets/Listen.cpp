//------------------------------------------------------------------------------
/// \file Listen.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref http://man7.org/linux/man-pages/man2/getsockname.2.html
//------------------------------------------------------------------------------
#include "Listen.h"

#include "IPC/Sockets/Socket.h"
#include "Utilities/ErrorHandling/ErrorNumber.h"

#include <optional>
#include <sys/socket.h> // ::getsockname

using Utilities::ErrorHandling::ErrorNumber;

namespace IPC
{
namespace Sockets
{

MakeListen::MakeListen(const int backlog_length):
  backlog_length_{backlog_length}
{}

std::optional<ErrorNumber> MakeListen::operator()(const Socket& socket)
{
  const int return_value {::listen(socket.fd(), backlog_length_)};

  return HandleListen()(return_value);
}

MakeListen::HandleListen::HandleListen() = default;

std::optional<ErrorNumber> MakeListen::HandleListen::operator()(
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

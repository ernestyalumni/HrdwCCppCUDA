//------------------------------------------------------------------------------
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref https://linux.die.net/man/2/socket
//------------------------------------------------------------------------------
#include "CreateSocket.h"

#include "Cpp/Utilities/TypeSupport/UnderlyingTypes.h"
#include "IPC/Sockets/ParameterFamilies.h"
#include "Utilities/ErrorHandling/HandleClose.h"
#include "Utilities/ErrorHandling/HandleReturnValue.h"

#include <optional>
#include <string>
#include <unistd.h> // ::close
#include <utility> // std::move

using Cpp::Utilities::TypeSupport::get_underlying_value;
using Utilities::ErrorHandling::HandleClose;
using Utilities::ErrorHandling::ThrowSystemErrorOnNegativeReturnValue;
using std::make_optional;
using std::move;
using std::nullopt;
using std::optional;
using std::to_string;

namespace IPC
{
namespace Sockets
{

SocketFd::SocketFd(
  const int domain,
  const int type,
  const int protocol,
  const int fd
  ):
  domain_{domain},
  type_{type},
  protocol_{protocol},
  fd_{fd}
{
  ThrowSystemErrorOnNegativeReturnValue handler {
    "Invalid input for fd:" + to_string(fd_)};

  handler(fd_);
}

SocketFd::SocketFd(SocketFd&& other):
  domain_{other.domain_},
  type_{other.type_},
  protocol_{other.protocol_},
  fd_{other.fd_}
{
  other.fd_ = -2;
}

SocketFd& SocketFd::operator=(SocketFd&& other)
{
  domain_ = other.domain_;
  type_ = other.type_;
  protocol_ = other.protocol_;
  fd_ = other.fd_;

  other.fd_ = -3;

  return *this;
}

SocketFd::~SocketFd()
{
  if (fd_ > -1)
  {
    HandleClose close_handler;

    const int close_return {::close(fd_)};
    close_handler(close_return);
  }
}

SocketFd SocketFd::extract_from_optional(optional<SocketFd>& optional_socket_fd)
{
  return move(*optional_socket_fd);
}

CreateSocket::CreateSocket(
  const int domain,
  const int type_value,
  const int protocol
  ):
  domain_{domain},
  type_value_{type_value},
  type_{from_type_value(type_value)},
  modifying_types_{},
  protocol_{protocol},
  socket_handle_{}
{}

CreateSocket::CreateSocket(
  const Domain domain,
  const Type type_enumeration,
  const int protocol
  ):
  domain_{to_domain_value(domain)},
  type_value_{to_type_value(type_enumeration)},
  type_{type_enumeration},
  modifying_types_{},
  protocol_{protocol},
  socket_handle_{}  
{}

void CreateSocket::add_behavior_modification(const BehaviorModifyingValue value)
{
  const auto result = modifying_types_.insert(value);
  if (result.second)
  {
    type_value_ = type_value_ | to_behavior_modifying_underlying_value(value);
  }
}

CreateSocket::OptionalSocketFd CreateSocket::operator()()
{
  const int return_value {::socket(domain_, type_value_, protocol_)};
  const auto socket_result = socket_handle_(return_value);

  // Error had occurred.
  if (static_cast<bool>(socket_result) || return_value < 0)
  {
    return nullopt;
  }

  SocketFd socket_fd {domain_, type_value_, protocol_, return_value};

  return make_optional<SocketFd>(move(socket_fd));
}

const int CreateSocket::to_domain_value(const Domain domain)
{
  return get_underlying_value<Domain>(domain);
}

const int CreateSocket::to_type_value(const Type type_enumeration)
{
  return get_underlying_value<Type>(type_enumeration);
}

const int CreateSocket::to_behavior_modifying_underlying_value(
  const BehaviorModifyingValue enumeration)
{
  return get_underlying_value<BehaviorModifyingValue>(enumeration);
}

} // namespace Sockets
} // namespace IPC

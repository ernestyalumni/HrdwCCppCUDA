//------------------------------------------------------------------------------
/// \file SetSocketOptions.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref https://linux.die.net/man/2/setsockopt
//------------------------------------------------------------------------------
#include "SetSocketOptions.h"

#include "Cpp/Utilities/TypeSupport/UnderlyingTypes.h"
#include "IPC/Sockets/ParameterFamilies.h"
#include "IPC/Sockets/Socket.h"
#include "Utilities/ErrorHandling/ErrorNumber.h"

#include <cassert>
#include <optional>
#include <sys/socket.h> // ::getsockname

using Cpp::Utilities::TypeSupport::get_underlying_value;
using Utilities::ErrorHandling::ErrorNumber;

namespace IPC
{
namespace Sockets
{

SetSocketOptions::SetSocketOptions(int level, int option_name, int optional_value):
 level_{level},
 option_name_{option_name},
 optional_value_{optional_value}
{
  assert(optional_value >= 0);
}

std::optional<ErrorNumber> SetSocketOptions::operator()(Socket& socket)
{
  const int return_value {
    ::setsockopt(
      socket.fd(),
      level_,
      option_name_,
      &optional_value_,
      sizeof(optional_value_))};
  
  return HandleSetSocketOptions()(return_value);
}

SetReusableSocketAddress::SetReusableSocketAddress():
  SetSocketOptions{
    get_underlying_value(Level::socket),
    get_underlying_value(Option::reuse_address),
    1}
{}

SetReusableAddressAndPort::SetReusableAddressAndPort():
  SetSocketOptions{
    get_underlying_value(Level::socket),
    get_underlying_value(Option::reuse_address) ||
      get_underlying_value(Option::reuse_port),
    1}
{}


} // namespace Sockets
} // namespace IPC

//------------------------------------------------------------------------------
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
//------------------------------------------------------------------------------
#include "CreateSocket.h"

#include "IPC/Sockets/CreateSocket.h"
#include "IPC/Sockets/ParameterFamilies.h"

namespace IPC
{
namespace Sockets
{
namespace TCP
{

CreateSocket::CreateSocket():
  IPC::Sockets::CreateSocket{Domain::ipv4, Type::stream}
{}

CreateSocket::CreateSocket(const Domain domain):
  IPC::Sockets::CreateSocket{domain, Type::stream}
{}

} // namespace TCP
} // namespace Sockets
} // namespace IPC

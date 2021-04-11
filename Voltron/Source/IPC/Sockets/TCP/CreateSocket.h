//------------------------------------------------------------------------------
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
//------------------------------------------------------------------------------
#ifndef IPC_SOCKETS_TCP_CREATE_SOCKET_H
#define IPC_SOCKETS_TCP_CREATE_SOCKET_H

#include "IPC/Sockets/CreateSocket.h"
#include "IPC/Sockets/ParameterFamilies.h"

namespace IPC
{
namespace Sockets
{
namespace TCP
{

class CreateSocket : public IPC::Sockets::CreateSocket
{
  public:

    CreateSocket();

    explicit CreateSocket(const Domain domain);

  private:
};

} // namespace TCP
} // namespace Sockets
} // namespace IPC

#endif // IPC_SOCKETS_TCP_CREATE_SOCKET_H

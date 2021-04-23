//------------------------------------------------------------------------------
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref https://linux.die.net/man/3/getaddrinfo
/// http://csapp.cs.cmu.edu/3e/ics3/code/src/csapp.c
//------------------------------------------------------------------------------
#include "GetReadyAddressInfo.h"

#include <cstring> // std::memset
#include <memory>
#include <netdb.h> // ::addrinfo
#include <string>
#include <sys/socket.h>
#include <sys/types.h>

namespace IPC
{

//GetReadyAddressInfo::GetReadyAddressInfo(const std::string& hostname):
//  hostname_node_ptr_{std::make_unique<char>(hostname)}

//------------------------------------------------------------------------------
/// \details
/// \ref https://linux.die.net/man/3/getaddrinfo
///
/// int ::getaddrinfo(const char*node, const char* service,
///   const struct addrinfo *hints, struct addrinfo **res);
//------------------------------------------------------------------------------


::addrinfo GetReadyAddressInfo::make_default_address_info()
{
  ::addrinfo address_info;
  // Specifying 0 indicates socket addresses of any type can be returned by
  // ::getaddrinfo.
  address_info.ai_socktype = 0;

  // ai_flags to (AI_V4MAPPED | AI_ADDRCONFIG) if ::addrinfo* specified as NULL
  //address_info.ai_flags = 0;

  // Specifying 0 indicates that socket addresses with any protocol can be
  // returned by getaddrinfo().
  address_info.ai_protocol = 0;

  return address_info;
}

void GetReadyAddressInfo::clear_address_info(::addrinfo& address_info)
{
  std::memset(&address_info, 0, sizeof(::addrinfo));
}


::addrinfo GetReadyAddressInfo::make_client_address_info()
{
  ::addrinfo address_info;

  GetReadyAddressInfo::clear_address_info(address_info);

  address_info.ai_socktype = SOCK_STREAM; // Open a connection.
  address_info.ai_flags = AI_NUMERICSERV; // ... using a numeric port argument.
  address_info.ai_flags |= AI_ADDRCONFIG; // Recommended for connections.

  return address_info;
}

::addrinfo GetReadyAddressInfo::make_server_address_info()
{
  ::addrinfo address_info;

  GetReadyAddressInfo::clear_address_info(address_info);

  address_info.ai_socktype = SOCK_STREAM; // Accept connections
  address_info.ai_flags = AI_PASSIVE | AI_ADDRCONFIG; // ... on any IP address
  address_info.ai_flags |= AI_NUMERICSERV; // ... using port number

  return address_info;
}

//void GetReadyAddressInfo::operator()()
//{
//  ::getaddrinfo()
//}

} // namespace IPC
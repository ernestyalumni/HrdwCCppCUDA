//------------------------------------------------------------------------------
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
//------------------------------------------------------------------------------
#ifndef IPC_GET_READY_ADDRESS_INFO_H
#define IPC_GET_READY_ADDRESS_INFO_H

#include <memory>
#include <netdb.h>
#include <string>

namespace IPC
{

class GetReadyAddressInfo
{
  public:

    GetReadyAddressInfo(const std::string& hostname);


    static ::addrinfo make_default_address_info();

    static void clear_address_info(::addrinfo& address_info);

    static ::addrinfo make_client_address_info();

    static ::addrinfo make_server_address_info();

    void operator()();

  private:

    std::unique_ptr<char> hostname_node_ptr_;

    ::addrinfo address_info_hints_;

    ::addrinfo socket_addresses_list_[];
};

} // namespace IPC

#endif // IPC_GET_READY_ADDRESS_INFO_H

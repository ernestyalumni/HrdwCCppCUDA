//------------------------------------------------------------------------------
/// \file UdpSocket.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  A UDP socket as RAII with a buffer, that's a std::array.
/// \ref      
/// \details A UDP socket as RAII with a buffer, that's a std::array. 
/// \copyright If you find this code useful, feel free to donate directly and
/// easily at this direct PayPal link: 
///
/// https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
/// 
/// which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.  
/// Otherwise, I receive emails and messages on how all my (free) material on
/// physics, math, and engineering have helped students with their studies, and
/// I know what it's like to not have money as a student, but love physics (or 
/// math, sciences, etc.), so I am committed to keeping all my material 
/// open-source and free, whether or not sufficiently crowdfunded, under the 
/// open-source MIT license: feel free to copy, edit, paste, make your own
/// versions, share, use as you wish.    
/// Peace out, never give up! -EY
//------------------------------------------------------------------------------
/// COMPILATION TIPS:
///  g++ -std=c++14 FileOpen_main.cpp FileOpen.cpp -o FileOpen_main
//------------------------------------------------------------------------------
#ifndef _UDPSOCKET_H_
#define _UDPSOCKET_H_

#include "../Socket.h"

#include <arpa/inet.h> // inet_aton
#include <array>
#include <cstddef> // std::size_t, for std::array
#include <cstring>
#include <string>

namespace Sockets
{

namespace Udp
{

//------------------------------------------------------------------------------
/// \details A DestinationSocketAddress is a SocketAddress. Thus, public
///   inheritance holds.
//------------------------------------------------------------------------------  
class DestinationSocketAddress : public Sockets::SocketAddress
{
  public:
    using Sockets::SocketAddress::SocketAddress;
    using Sockets::SocketAddress::get_sockaddr_in_uptr;
    using Sockets::SocketAddress::to_sockaddr_uptr;
    using Sockets::SocketAddress::size;
    using Sockets::SocketAddress::set_sockaddr_in_uptr;

    // Constructor, given only a port number
    explicit DestinationSocketAddress(const uint16_t sin_port = 0):
      SocketAddress(AF_INET, sin_port, INADDR_ANY)
    {}

    // 
    void Ipv4address_to_network_byte_order(const std::string str)
    {
      Ipv4address_to_network_byte_order(str.c_str());
    }

    // 
    void Ipv4address_to_network_byte_order(const char* cp)
    {
      const int result_of_inet_aton {
        ::inet_aton(cp, &(get_sockaddr_in_uptr()->sin_addr))};

      if (result_of_inet_aton == 0)
      {
        throw std::runtime_error("inet_aton() failed\n");
      }
    }
};

//------------------------------------------------------------------------------
/// \details A UdpSocket is a Socket. Thus, public inheritance holds.
//------------------------------------------------------------------------------  
template <std::size_t buffer_length>
class UdpSocket : public Sockets::Socket
{
  public:
    using Sockets::Socket::Socket;
    using Sockets::Socket::bind;
    using Sockets::Socket::get_socket_name;

    void sendto(DestinationSocketAddress& destination_socket_address) const
    {
      ssize_t sendto_result {
        ::sendto(
          fd(),
          buffer_.data(), buffer_.size(),
          0,
          destination_socket_address.to_sockaddr_uptr().get(),
          destination_socket_address.size())
      };
      if (sendto_result < 0)
      {
        throw std::runtime_error("sendto failed\n");
      }
    }

    // Accessors
    std::array<char, buffer_length> buffer() const
    {
      return buffer_;
    }

    // Setters
    void set_buffer(std::string& string_message)
    {
      std::strcpy(buffer_.data(), string_message.c_str());
    }

  protected:

    using Sockets::Socket::fd;

  private:

    std::array<char, buffer_length> buffer_;
};

} // Udp

} // Sockets

#endif // _UDPSOCKET_H_

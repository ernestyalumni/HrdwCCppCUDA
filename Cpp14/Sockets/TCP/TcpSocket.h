//------------------------------------------------------------------------------
/// \file TcpSocket.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  A TCP/IP socket as RAII with a buffer, that's a std::array.
/// \ref https://www.binarytides.com/server-client-example-c-sockets-linux/     
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
#ifndef _TCPSOCKET_H_
#define _UDPSOCKET_H_

#include "../Socket.h"

#include <arpa/inet.h> // inet_aton
#include <array>
#include <cstddef> // std::size_t, for std::array
#include <cstring>
#include <string>
#include <sys/socket.h> // ::listen, ::accept

namespace Sockets
{

namespace Tcp
{

//------------------------------------------------------------------------------
/// \details A UdpSocket is a Socket. Thus, public inheritance holds.
//------------------------------------------------------------------------------  
//template <std::size_t buffer_length>
class TcpSocket : public Sockets::SocketV2
{
  public:

    explicit TcpSocket(
      const int domain = AF_INET,
      const int type = SOCK_STREAM,
      const int protocol = 0,
      const uint16_t sin_port = 0,
      uint32_t s_address = INADDR_ANY):
      SocketV2(domain, type, protocol, sin_port, s_address)      
    {}

    explicit TcpSocket(
      Sockets::SocketAddressIn& socket_address_in,
      const int domain = AF_INET,
      const int type = SOCK_STREAM,
      const int protocol = 0):
      SocketV2(socket_address_in, domain, type, protocol)      
    {}

    using Sockets::SocketV2::bind;
    using Sockets::SocketV2::get_socket_name;
    using Sockets::SocketV2::get_socket_length_ptr;

    //--------------------------------------------------------------------------
    /// \brief Class wrapper function for ::setsockopt
    ///
    /// \details We use ::setsockopt to set SO_REUSEADDR. This allows us to 
    /// reuse the port immediately as soon as the service exits. 
    /// Some operating systems will not allow immeidate reuse on the change that
    /// osome packets may still be en route to the port.
    //--------------------------------------------------------------------------
    void set_socket_options(
      )

    //--------------------------------------------------------------------------
    /// \brief Class wrapper function for ::listen
    //--------------------------------------------------------------------------
    void listen(int sockfd, int backlog)
    {
      if (::listen(sockfd, backlog) < 0)
      {
        throw std::runtime_error("listen failed.");
      }
    }

    //--------------------------------------------------------------------------
    /// \brief Class wrapper function for ::accept
    //--------------------------------------------------------------------------
    void accept(int sockfd, Sockets::SocketAddressIn& socket_address_in)
    {
      if (::accept(
        sockfd,
        socket_address_in.to_sockaddr(),
        get_socket_length_ptr()) < 0)
      {
        throw std::runtime_error("accept failed.");
      }
    }

    //--------------------------------------------------------------------------
    /// \brief Class wrapper function for ::recv
    //--------------------------------------------------------------------------
    void receive(int sockfd, 

      int flags)
    {

    }

    // Accessors
#if 0
    std::array<char, buffer_length> buffer() const
    {
      return buffer_;
    }

    // Setters
    void set_buffer(std::string& string_message)
    {
      std::strcpy(buffer_.data(), string_message.c_str());
    }
#endif 


  protected:

    using Sockets::SocketV2::fd;
    using Sockets::SocketV2::socket_address_in;

//  private:

  //  std::array<char, buffer_length> buffer_;
};

} // Tcp

} // Sockets

#endif // _TCPSOCKET_H_


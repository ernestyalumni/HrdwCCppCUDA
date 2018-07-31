//------------------------------------------------------------------------------
/// \file UDPSocket.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  A UDP socket as RAII 
/// \ref      
/// \details Using RAII for UDP sockets. 
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
///  g++ -std=c++14 Socket_main.cpp -o Socket_main
//------------------------------------------------------------------------------
#ifndef _UDP_SOCKET_H_
#define _UDP_SOCKET_H_

#include "../Socket.h" // AllDomains, AllTypes, AllSpecialAddresses

#include <array>
#include <cerrno> // errno
#include <cstddef> // std::size_t
#include <cstring> // std::memcpy
#include <system_error>
#include <sys/socket.h> // ::sendto, ::recvfrom

namespace IPC
{

namespace Sockets
{


template <std::size_t N_READ, std::size_t N_WRITE>
class UdpSocket : public Socket<
  AllDomains::ipv4_internet,
  static_cast<int>(AllTypes::datagram)
  >
{
  public:

    using UdpSocketType = Socket<
      AllDomains::ipv4_internet,
      static_cast<int>(AllTypes::datagram)>;

    using UdpSocketType::Socket;

    explicit UdpSocket(const uint16_t port, const bool is_server):
      UdpSocketType{port, static_cast<uint32_t>(AllSpecialAddresses::any)}
    {}

    explicit UdpSocket(
      const uint16_t port,
      uint32_t s_address = static_cast<uint32_t>(AllSpecialAddresses::any)):
        remote_address_in_{
          static_cast<uint16_t>(AllDomains::ipv4_internet),
          port,
          s_address}
    {}

    //--------------------------------------------------------------------------
    /// Wrappers for ::sendto, ::recvfrom
    //--------------------------------------------------------------------------

    ssize_t send_to(const char* message, size_t length)
    {
      ssize_t send_to_result{
        ::sendto(
          fd(),
          message,
          length,
          0,
          remote_address_in().to_sockaddr(), 
          socket_address_in().size())};

      if (send_to_result < 0)
      {
        // continue, but take note of the error:
        std::cout << " errno : " << std::strerror(errno) << '\n';
        throw std::system_error(
          errno,
          std::generic_category(),
          "Failed to close file descriptor (::close) \n");
      }  
      return send_to_result;    
    }

    template <std::size_t n>
    ssize_t send_to(const std::array<char, n>& message)
    {
      return send_to(message.data(), n);
    }

    ssize_t send_to()
    {
      return send_to(read_buffer_.data(), N_READ);
    }

    ssize_t receive_from(char* message, size_t length)
    {
      ssize_t receive_from_result{
        ::recvfrom(
          fd(),
          message,
          length,
          0,
          remote_address_in().to_sockaddr(), 
          get_socket_length_ptr())};

      if (receive_from_result < 0)
      {
        // continue, but take note of the error:
        std::cout << " errno : " << std::strerror(errno) << '\n';
        throw std::system_error(
          errno,
          std::generic_category(),
          "Failed to receive message (::recvfrom) \n");
      }  
      return receive_from_result;    
    }

    template <std::size_t m>
    ssize_t receive_from(std::array<char, m>& message)
    {
      return receive_from(message.data(), m);
    }

    ssize_t receive_from()
    {
      return receive_from(write_buffer_.data(), N_WRITE);
    }

  protected:

    using UdpSocketType::fd;
    using UdpSocketType::socket_address_in;
    using UdpSocketType::get_socket_length_ptr;
    using UdpSocketType::close;

    // Accessors

    SocketAddressIn& remote_address_in()
    {
      return remote_address_in_;
    }

    std::array<char, N_READ> read_buffer() const
    {
      return read_buffer_;
    }

    std::array<char, N_READ> write_buffer() const
    {
      return write_buffer_;
    }

    // Setters
    void set_write_buffer(const char* message, std::size_t count) 
    {
      std::memcpy(write_buffer_.data(), message, count);
    }


  private:

    std::array<char, N_READ> read_buffer_;
    std::array<char, N_WRITE> write_buffer_;

    SocketAddressIn remote_address_in_; // remote address
};


template <std::size_t M_READ, std::size_t M_WRITE>
class UdpServer : public UdpSocket<M_READ, M_WRITE>
{
  public:

    using UdpSocket<M_READ, M_WRITE>::UdpSocket;

    // Will bind socket to any valid IP address, and a specific port
    explicit UdpServer(const uint16_t port):
      UdpSocket<M_READ, M_WRITE>{port, true}
    {}
};

template <std::size_t M_READ, std::size_t M_WRITE>
class UdpClient : public UdpSocket<M_READ, M_WRITE>
{
  public:

    using UdpSocket<M_READ, M_WRITE>::UdpSocket;
    using UdpSocket<M_READ, M_WRITE>::set_write_buffer;
    using UdpSocket<M_READ, M_WRITE>::write_buffer;

    explicit UdpClient(const uint16_t port):
      UdpSocket<M_READ, M_WRITE>{
        port,
        static_cast<uint32_t>(AllSpecialAddresses::any)}
    {}
};


} // namespace Sockets

} // namespace IPC

#endif // _UDP_SOCKET_H_

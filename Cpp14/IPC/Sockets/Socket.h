//------------------------------------------------------------------------------
/// \file Socket.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  A socket as RAII 
/// \ref      
/// \details Using RAII for socket. 
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
#ifndef _SOCKET_H_
#define _SOCKET_H_

// \ref http://pubs.opengroup.org/onlinepubs/7908799/xsh/unistd.h.html
// \ref http://pubs.opengroup.org/onlinepubs/7908799/xns/netinetin.h.html
#include <arpa/inet.h> // htonl, htons
#include <cerrno> // errno
#include <cstring> // strerror, std::memset
#include <iostream>
#include <memory>
#include <netinet/in.h> // ::sockaddr_in
#include <stdexcept> // std::runtime_error
#include <sys/socket.h> // AF_INET,AF_UNIX,AF_LOCAL,etc.; communication domains
#include <system_error>
#include <unistd.h> // ::close, ::sleep, ::unlink

namespace IPC
{

namespace Sockets
{

//------------------------------------------------------------------------------
/// \brief enum class for all socket domains (communication domains)
/// \details Selects protocol family which will be used for communication.
///   These families are defined in <sys.socket.h>
/// \ref https://linux.die.net/man/2/socket
//------------------------------------------------------------------------------
enum class AllDomains : int
{
  unix = AF_UNIX,
  local = AF_LOCAL,
  ipv4_internet = AF_INET,
  ipv6_internet = AF_INET6,
  ipx_Novell = AF_IPX,
  kernel_user_interface = AF_NETLINK, // Kernel user interface device
  itu_t_x25 = AF_X25, // ITU-T X.25 / ISO-8208 protocol
  amateur_radio = AF_AX25, // Amateur radio AX.25 protocol
  raw_ATM_PVCs = AF_ATMPVC, // Access to raw ATM PVCs
  appletalk = AF_APPLETALK, // AppleTalk
  packet = AF_PACKET, // low level packet interface 
  crypto = AF_ALG // Interface to kernel crypto API
};

enum class AllTypes : int
{
  stream = SOCK_STREAM,
  datagram = SOCK_DGRAM,
  sequenced_packets = SOCK_SEQPACKET,
  raw = SOCK_RAW,
  reliable_datagram_layer = SOCK_RDM,
  nonblocking = SOCK_NONBLOCK, 
  close_on_exec = SOCK_CLOEXEC
};

enum class AllSpecialAddresses : uint32_t
{
  any = INADDR_ANY, 
  loopback = INADDR_LOOPBACK, // (127.0.0.1), always refers to local host via
  // loopback device
  broadcast = INADDR_BROADCAST // (255.255.255.255), any host and has same
  // effect on bind as INADDR_ANY
};

//------------------------------------------------------------------------------
/// \brief Derived class for ::sockaddr_in
///
/// \details INADDR_ANY is the IP address and 0 is the socket (port).
/// ::htonl converts a long integer (e.g. address) to a network representation
/// (IP-standard byte ordering)
/// 
/// data members of ::sockaddr_in (and thus SocketAddressIn), in order:
/// struct sockaddr_in
/// {
///   sa_family_t sin_family;
///   in_port_t sin_port;
///   struct in_addr sin_addr; 
/// }
/// sin-family is always set to AF_INET. This is required. 
/// cf. https://linux.die.net/man/7/ip "Address format" 
//------------------------------------------------------------------------------
class SocketAddressIn: public ::sockaddr_in
{
  public:

    explicit SocketAddressIn(
      const uint16_t sin_family = AF_INET,
      const uint16_t sin_port = 0,
      const uint32_t s_address = INADDR_ANY):
      ::sockaddr_in {
        sin_family,
        ::htons(sin_port),
        ::in_addr{::htonl(s_address)}}
    {}

    explicit SocketAddressIn(const uint16_t sin_port):
      SocketAddressIn {AF_INET, ::htons(sin_port), INADDR_ANY}
    {}

    const unsigned int size() const 
    {
      return sizeof(::sockaddr_in);
    }

    ::sockaddr* to_sockaddr()
    {
      return reinterpret_cast<::sockaddr*>(this);
    }

    //--------------------------------------------------------------------------
    /// \brief Wrapper for memset.
    /// \ref https://en.cppreference.com/w/cpp/string/byte/memset
    //--------------------------------------------------------------------------
    void set_to_zero()
    {
      std::memset(this, 0, size());
    }
};

//------------------------------------------------------------------------------
/// \brief Socket class template (usually for IP Socket).
/// \details Create a IP Socket; can be a UDP Socket or TCP/IP Socket.
///   Request the Internet address protocol. 
///   (It could be) a datagram interface (UDP/IP), or socket stream.
/// \ref https://www.cs.rutgers.edu/~pxk/417/notes/sockets/demo-udp-01.html
//------------------------------------------------------------------------------
template <
  AllDomains domain,// = AllDomains::ipv4_internet,
  int type,
  int protocol = 0>
class Socket
{
  public:

    // Will bind to all local addresses, and pick any port number (port 0)
    Socket():
      fd_{::socket(static_cast<uint16_t>(domain), type, protocol)},
      socket_address_in_{
        static_cast<uint16_t>(domain),
        0,
        static_cast<uint32_t>(AllSpecialAddresses::any)}
    {
      check_fd();
    }

    explicit Socket(
      const uint16_t sin_port = 0,
      uint32_t s_address = static_cast<uint32_t>(AllSpecialAddresses::any)):
        fd_{::socket(static_cast<uint16_t>(domain), type, protocol)},
        socket_address_in_{static_cast<uint16_t>(domain), sin_port, s_address}
    {
      check_fd();
    }

    explicit Socket(SocketAddressIn& socket_address_in):
      fd_{::socket(static_cast<uint16_t>(domain), type, protocol)},
      socket_address_in_{std::move(socket_address_in)}
    {
      check_fd();
    }

    ~Socket()
    {
      if (::close(fd_) < 0)
      {
//        throw std::runtime_error( // throw will always call terminate; 
        // dtors default to noexcept
          std::cout << "Failed to close file descriptor, fd_ (::close)\n";
      }
    }

    void bind()
    {
      if (::bind(fd_, 
            socket_address_in_.to_sockaddr(),
            socket_address_in_.size()) < 0)
      {
        throw std::runtime_error("bind failed");
      }
    }

    int get_socket_name()
    {
      const int get_socket_name_result {
        ::getsockname(
          fd_,
          socket_address_in_.to_sockaddr(),
          socket_length_uptr_.get()
        )
      };
      if (get_socket_name_result < 0)
      {
        throw std::runtime_error("getsockname failed");
      }
      else
      {
        return get_socket_name_result;
      } 
    }   

  protected:

    // protected, so one wouldn't be able to do something like ::close(fd)
    const int fd() const
    {
      return fd_;
    }

    SocketAddressIn& socket_address_in()
    {
      return socket_address_in_;
    }

    socklen_t* get_socket_length_ptr() const
    {
      return socket_length_uptr_.get();
    }

    int close()
    {
      const int close_result {::close(fd_)};
      if (close_result < 0)
      {
        std::cout << " errno : " << std::strerror(errno) << '\n';
        throw std::system_error(
          errno,
          std::generic_category(),
          "Failed to close file descriptor (::close) \n");
      }
      return close_result;
    }

  private:

    void check_fd()
    {
      if (fd_ < 0)
      {
        throw std::runtime_error("socket construction failed");
      }
      return;
    }

    int fd_;

    SocketAddressIn socket_address_in_;

    std::unique_ptr<socklen_t> socket_length_uptr_ {
      std::make_unique<socklen_t>(sizeof(::sockaddr_in))};
};



} // namespace Sockets

} // namespace IPC

#endif // _SOCKET_H_

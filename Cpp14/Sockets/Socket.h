//------------------------------------------------------------------------------
/// \file Socket.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  A socket as RAII 
/// \ref      
/// \details Using RAII for socket. 
/// \copyright If you find this code useful, feel free to donate directly and easily at 
/// this direct PayPal link: 
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
#include <netinet/in.h> // ::sockaddr_in
#include <stdexcept> // std::runtime_error
#include <sys/socket.h> // AF_INET
#include <unistd.h> // ::close, ::sleep, ::unlink
#include <memory>

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
  kernel_user_interface = AF_NETLINK,
  itu_t_x25 = AF_X25,
  amateur_radio = AF_AX25,
  raw_ATM_PVCs = AF_ATMPVC,
  appletalk = AF_APPLETALK,
  packet = AF_PACKET // low level packet interface 
};

//------------------------------------------------------------------------------
/// \brief enum class for "common" socket domains (communication domains)
/// \details Selects protocol family which will be used for communication.
///   These families are defined in <sys.socket.h>
///   Only the "most commonly used" communication domains are in this. 
/// \ref https://linux.die.net/man/2/socket
//------------------------------------------------------------------------------
enum class CommonDomains : int
{
  unix = AF_UNIX,
  local = AF_LOCAL,
  ipv4_internet = AF_INET,
  packet = AF_PACKET // low level packet interface 
};

//------------------------------------------------------------------------------
/// \brief RAII class for ::sockaddr_in
///
/// \details INADDR_ANY is the IP address and 0 is the socket (port).
/// ::htonl converts a long integer (e.g. address) to a network representation
/// (IP-standard byte ordering)
//------------------------------------------------------------------------------
class SocketAddress
{
  public:

    // Constructor matching low-level API for ::sockaddr_in
    explicit SocketAddress(
      const uint16_t sin_family = AF_INET,
      const uint16_t sin_port = 0,
      const uint32_t s_address = INADDR_ANY):
      sin_family_{sin_family},
      sin_port_{::htons(sin_port)},
      in_addr_{::htonl(s_address)},
      sockaddr_in_{::sockaddr_in{sin_family_, sin_port_, in_addr_}},
      sockaddr_in_uptr_{std::make_unique<::sockaddr_in>(sockaddr_in_)},
      socket_length_uptr_{
        std::make_unique<socklen_t>(sizeof(::sockaddr_in))}
    {}

    // Accessors
    auto get_sockaddr_in_uptr()
    {
      return sockaddr_in_uptr_.get();
    }

    socklen_t* get_socket_length_ptr() const
    {
      return socket_length_uptr_.get();
    }

    std::unique_ptr<::sockaddr> to_sockaddr_uptr() const
    {
      return 
        std::make_unique<::sockaddr>(
          *(reinterpret_cast<::sockaddr*>(
            sockaddr_in_uptr_.get()))
        );
    }

    const unsigned int size() const 
    {
      return sizeof(::sockaddr_in);
    }

    // Setters
    void set_sockaddr_in_uptr(std::unique_ptr<::sockaddr_in> sockaddr_in_uptr)
    {
      sockaddr_in_uptr_ = std::move(sockaddr_in_uptr);
    }

  protected:

    ::sockaddr_in sockaddr_in() const
    {
      return sockaddr_in_;
    }

    // Accessors
    std::unique_ptr<::sockaddr_in> sockaddr_in_uptr()
    {
      return std::move(sockaddr_in_uptr_);
    }

    std::unique_ptr<socklen_t> socket_length_uptr()
    {
      return std::move(socket_length_uptr_);
    }

  private:

    // \ref http://www.ccplusplus.com/2011/10/struct-sockaddrin.html

    // sa_family_t = uint16_T
    uint16_t sin_family_; // Address family (communication domain)
    uint16_t sin_port_;   // Port number
    ::in_addr in_addr_;   // Internet address

    ::sockaddr_in sockaddr_in_;
    std::unique_ptr<::sockaddr_in> sockaddr_in_uptr_;

    std::unique_ptr<socklen_t> socket_length_uptr_;
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
//------------------------------------------------------------------------------
class SocketAddressIn: public ::sockaddr_in
{
  public:

    explicit SocketAddressIn(
      const uint16_t sin_family = AF_INET,
      const uint16_t sin_port = 0,
      const uint32_t s_address = INADDR_ANY):
      ::sockaddr_in {sin_family, ::htons(sin_port), ::htonl(s_address)}
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
};

//------------------------------------------------------------------------------
/// \brief Socket class (usually for IP Socket)
/// \details Create a IP Socket; can be a UDP/IP Socket.
///   Request the Internet address protocol. 
///   (It could be) a datagram interface (UDP/IP)
/// \ref https://www.cs.rutgers.edu/~pxk/417/notes/sockets/demo-udp-01.html
//------------------------------------------------------------------------------
class Socket
{
  public:

    explicit Socket(const int domain = AF_INET, const int type = SOCK_DGRAM,
      const int protocol = 0, uint32_t s_address = INADDR_ANY):
      domain_{domain}, type_{type}, protocol_{protocol},
      fd_{::socket(domain, type, protocol)},
      socket_address_{static_cast<uint16_t>(domain), 0, s_address}
    {
      check_fd();
    }

    explicit Socket(SocketAddress& socket_address, 
      const int domain = AF_INET, const int type = SOCK_DGRAM,
      const int protocol = 0):
      domain_{domain}, type_{type}, protocol_{protocol},
      fd_{::socket(domain, type, protocol)},
      socket_address_{std::move(socket_address)}
    {
      check_fd();
    }

    ~Socket()
    {
      ::close(fd_);
    }

    void bind()
    {
      if (::bind(
        fd_, 
        socket_address_.to_sockaddr_uptr().get(),
        socket_address_.size()
        ) < 0)
      {
        throw std::runtime_error("bind failed");
      }
    }

    int get_socket_name() const
    {
      const int get_socket_name_result {
        ::getsockname(
          fd_,
          socket_address_.to_sockaddr_uptr().get(),
          socket_address_.get_socket_length_ptr()
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

    // Accessors
    const int domain() const
    {
      return domain_;
    }

    const int type() const
    {
      return type_;
    }

    const int protocol() const
    {
      return protocol_;
    }

  protected:

    // protected, so one wouldn't be able to do something like ::close(fd)
    const int fd() const
    {
      return fd_;
    }

  private:
    int domain_;
    int type_;
    int protocol_;

    int fd_;

    SocketAddress socket_address_;

    void check_fd()
    {
      if (fd_ < 0)
      {
        throw std::runtime_error("socket construction failed");
      }
      return;
    }
};

//------------------------------------------------------------------------------
/// \brief Socket class (usually for IP Socket), version 2
/// \details Create a IP Socket; can be a UDP Socket or TCP/IP Socket.
///   Request the Internet address protocol. 
///   (It could be) a datagram interface (UDP/IP), or socket stream.
/// \ref https://www.cs.rutgers.edu/~pxk/417/notes/sockets/demo-udp-01.html
//------------------------------------------------------------------------------
class SocketV2
{
  public:

    explicit SocketV2(
      const int domain,
      const int type,
      const int protocol = 0,
      const uint16_t sin_port = 0,
      uint32_t s_address = INADDR_ANY):
      fd_{::socket(domain, type, protocol)},
      socket_address_in_{static_cast<uint16_t>(domain), sin_port, s_address}
    {
      check_fd();
    }

    explicit SocketV2(
      SocketAddressIn& socket_address_in, 
      const int domain, const int type, const int protocol = 0):
      fd_{::socket(domain, type, protocol)},
      socket_address_in_{std::move(socket_address_in)}
    {
      check_fd();
    }

    ~SocketV2()
    {
      ::close(fd_);
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

#endif // _SOCKET_H_

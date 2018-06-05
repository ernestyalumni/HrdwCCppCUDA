//------------------------------------------------------------------------------
/// \file SocketAddress_playground.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  A socket address struct playground  
/// \ref https://www.cs.rutgers.edu/~pxk/417/notes/sockets/udp.html     
/// \details Playing around with the socket address struct. 
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

//#include <sys/socket.h>

#include <arpa/inet.h> // htonl, htons
#include <iostream>
#include <memory>

#include <netinet/in.h> // ::sockaddr_in

#include <cstring> // std::memset

//------------------------------------------------------------------------------
/// \ref https://linux.die.net/man/3/htons
/// \details htons converts unsigned short integer hostshort from network byte 
///   order to host byte order (uint16_t htons(uint16_t hostshort))
/// htonl converts unsigned integer hostlong from host byte order to network 
///   byte order.
//------------------------------------------------------------------------------

class SocketAddress : public ::sockaddr_in
{
  public:

    // sin_port, s_addr_in
    explicit SocketAddress(
      const uint16_t sin_family = AF_INET,
      const uint16_t sin_port = 0,
      const uint32_t s_address = INADDR_ANY):
      sin_family_{sin_family},
      sin_port_{::htons(sin_port)},
      in_addr_{::htonl(s_address)},
      sockaddr_in_{::sockaddr_in{sin_family_, sin_port_, in_addr_}},
      sockaddr_in_uptr_{std::make_unique<::sockaddr_in>(sockaddr_in_)}
    {
      #if 0
      std::memset((void *)sockaddr_in_ptr_, 0, sizeof(*sockaddr_in_ptr_));
      sockaddr_in_ptr_->sin_family = sin_family;
      sockaddr_in_ptr_->sin_port = ::htons(sin_port);
      sockaddr_in_ptr_->sin_addr.s_addr = ::htonl(s_address);
      #endif 
    }

    // Accessors

    ::sockaddr_in sockaddr_in() const
    {
      return sockaddr_in_;
    }

    std::unique_ptr<::sockaddr_in> sockaddr_in_uptr()
    {
      return std::move(sockaddr_in_uptr_);
    }

    auto get_sockaddr_in()
    {
      return sockaddr_in_uptr_.get();
    }

    // Setters
    void set_sockaddr_in_uptr(std::unique_ptr<::sockaddr_in> sockaddr_in_uptr)
    {
      sockaddr_in_uptr_ = std::move(sockaddr_in_uptr);
    }

  private:

    // \ref http://www.ccplusplus.com/2011/10/struct-sockaddrin.html

    // sa_family_t = uint16_T
    uint16_t sin_family_; // Address family (communication domain)
    uint16_t sin_port_;   // Port number
    ::in_addr in_addr_;   // Internet address

    ::sockaddr_in sockaddr_in_;
    std::unique_ptr<::sockaddr_in> sockaddr_in_uptr_;
//    ::sockaddr_in* sockaddr_in_ptr_;
};


int main()
{
  // \ref http://www.ccplusplus.com/2011/10/struct-sockaddrin.html

  ::in_addr in_addr0 {INADDR_ANY};

  SocketAddress socket_address {AF_INET, 0, INADDR_ANY};

  std::cout << socket_address.sockaddr_in().sin_family << ' ' << 
    socket_address.sockaddr_in().sin_port << ' ' << 
    socket_address.sockaddr_in().sin_addr.s_addr << '\n';

  std::unique_ptr<::sockaddr_in> sockaddr_in_uptr0 = 
    socket_address.sockaddr_in_uptr();

  std::cout << (sockaddr_in_uptr0 == nullptr) << '\n'; // 1

  std::unique_ptr<::sockaddr_in> sockaddr_in_uptr1 = 
    socket_address.sockaddr_in_uptr();

  std::cout << (sockaddr_in_uptr1 == nullptr) << '\n'; // 0 

  socket_address.set_sockaddr_in_uptr(std::move(sockaddr_in_uptr0));



}

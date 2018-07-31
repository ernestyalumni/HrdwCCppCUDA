//------------------------------------------------------------------------------
/// \file Simple_UDP_client_send_main.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  A simple UDP client.
/// \ref      
/// \details Using RAII for UDP sockets client. 
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
///  g++ -std=c++14 Simple_UDP_client_send_main.cpp -o Simple_UDP_client_send_main
//------------------------------------------------------------------------------

#include "UDPSocket.h"

#include <cstddef> // std::size_t
#include <string>

constexpr std::size_t BUFFER_LENGTH {100};
constexpr uint16_t PORT {21234};
const std::string hello_world {"hello world"};

using IPC::Sockets::UdpClient;

int main()
{
  UdpClient<BUFFER_LENGTH, BUFFER_LENGTH> udp_client {PORT};

  // bind it to all local addresses and pick any port number
  udp_client.bind();

  udp_client.set_write_buffer(hello_world.data(), hello_world.size());
}

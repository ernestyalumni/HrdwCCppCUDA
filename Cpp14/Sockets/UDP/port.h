//------------------------------------------------------------------------------
/// \file port.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  SERVICE_PORT defines the default port number for this service.
/// \ref Paul Krzyzanowski. 
/// \details Replace the number with a unique number > 1024. 
///   A reasonable number maybe obtained from, say, 4 digits of your id + 1024. 
/// \copyright If you find this code useful, feel free to donate directly and 
///   easily at this direct PayPal link: 
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

#ifndef _PORT_H_
#define _PORT_H_

namespace Sockets
{

namespace Udp
{

constexpr const uint16_t SERVICE_PORT {21234}; // hard-coded port number

} // Udp

} // Sockets

#endif // _UDPSOCKET_H_

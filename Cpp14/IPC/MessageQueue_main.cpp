//------------------------------------------------------------------------------
/// \file MessageQueue_main.cpp
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
#include "MessageQueue/MessageQueue.h"

#include <iostream>
#include <mqueue.h>

using IPC::AllFlags;
using IPC::MessageAttributes;
using IPC::maximum_message_size;
using IPC::maximum_number_of_messages;

int main()
{
  // AllFlagsRepresentsAllFlags
  std::cout << "\n AllFlagsRepresentsAllFlags \n";  
  std::cout << " AllFlags::receive_only : " << 
    static_cast<long>(AllFlags::receive_only) << '\n'; // 0
  std::cout << " AllFlags::send_only : " << 
    static_cast<long>(AllFlags::send_only) << '\n'; // 1
  std::cout << " AllFlags::send_and_receive : " << 
    static_cast<long>(AllFlags::send_and_receive) << '\n'; // 2 
  std::cout << " AllFlags::close_on_execution : " << 
    static_cast<long>(AllFlags::close_on_execution) << '\n'; // 524288
  std::cout << " AllFlags::create : " << 
    static_cast<long>(AllFlags::create) << '\n'; // 64
  std::cout << " AllFlags::exclusive_existence : " << 
    static_cast<long>(AllFlags::exclusive_existence) << '\n'; // 128
  std::cout << " AllFlags::nonblocking : " << 
    static_cast<long>(AllFlags::nonblocking) << '\n'; // 2048


  // ::mq_attrConstructs
  std::cout << "\n ::mq_attrConstructs\n";

  ::mq_attr test_mq_attr1 {};
  ::mq_attr test_mq_attr2 {
    static_cast<long>(AllFlags::send_and_receive) | 
      static_cast<long>(AllFlags::create) |
      static_cast<long>(AllFlags::exclusive_existence) |
      static_cast<long>(AllFlags::nonblocking),
    maximum_number_of_messages,
    maximum_message_size};
  std::cout << test_mq_attr1.mq_flags << ' ' << test_mq_attr1.mq_maxmsg << 
    ' ' << test_mq_attr1.mq_msgsize << ' ' << test_mq_attr1.mq_curmsgs << '\n';
  std::cout << test_mq_attr2.mq_flags << ' ' << test_mq_attr2.mq_maxmsg << 
    ' ' << test_mq_attr2.mq_msgsize << ' ' << test_mq_attr2.mq_curmsgs << '\n';

  // MessageAttributesConstructs
  std::cout << "\n MessageAttributesConstructs\n";
  MessageAttributes message_attributes {};
  std::cout << message_attributes.mq_flags << ' ' <<
    message_attributes.mq_maxmsg << ' ' << message_attributes.mq_msgsize <<
    ' ' << message_attributes.mq_curmsgs << '\n';


}


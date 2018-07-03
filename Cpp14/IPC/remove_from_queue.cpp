//------------------------------------------------------------------------------
/// \file remove_from_queue.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Simply request a message from a queue, and displays queue
///   attributes. 
/// \ref      
/// \details Simply request a message from a queue, and displays queue
///   attributes.
///   If you execute the program a second time, you will see the program hang,
///   because the process is suspended until another message is added to the 
///   queue. 
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
///  g++ -std=c++14 -lrt -Wall add_to_queue.cpp -o add_to_queue
///  g++ -std=c++14 -lrt remove_from_queue.cpp -o remove_from_queue
/// ./add_to_queue
/// ./remove_from_queue
//------------------------------------------------------------------------------
#include "MessageQueue/MessageQueue.h"

#include <array>
#include <iostream>
#include <string>
#include <sys/types.h> // pid_t
#include <unistd.h> // ::getpid
#include <ctime> // std::time_t, std::ctime

using IPC::AllFlags;
using IPC::AllModes;
using IPC::MessageAttributes;
using IPC::MessageQueue;
using IPC::maximum_number_of_messages;

// name of the POSIX object referencing the queue
const std::string message_queue_object_name {"/myqueue123"};

// max length of a message (just for this process)
constexpr long maximum_message_length {10000};

int main()
{

  std::array<char, maximum_message_length> message_content;

  MessageAttributes attributes {
    0,
    maximum_number_of_messages,
    maximum_message_length};

  std::cout << " attributes, initially : " << attributes << '\n';

  // opening the queue -- ::mq_open()
  // getting the attributes from the queue -- mq_getattr()
  MessageQueue message_queue {
    message_queue_object_name.c_str(),
    static_cast<long>(AllFlags::send_and_receive),
    666,
    attributes};

  std::cout << " Queue : " << message_queue_object_name <<
    " \n\t - stores at most " << 
    message_queue.attributes().mq_maxmsg <<
    " messages\n\t - large at most " << 
    message_queue.attributes().mq_msgsize << 
    " bytes each\n\t - current holds " <<
    message_queue.attributes().mq_curmsgs << " messages. \n";

  std::cout << "\n message_queue : " << message_queue << '\n';

  // getting a message
  const ssize_t remove_from_queue_result {
    message_queue.remove_from_queue(
    message_content.data(),
    maximum_message_length)};

  std::cout << "Received message (" << remove_from_queue_result <<
    " bytes) from " << message_queue.priority() <<
    " : " << std::string{message_content.data()} << '\n';

  std::cout << "\n message_queue after receive : " << message_queue << '\n';

  message_queue.unlink();
}

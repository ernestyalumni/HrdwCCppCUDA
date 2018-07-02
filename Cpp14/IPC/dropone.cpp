//------------------------------------------------------------------------------
/// \file dropeone.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Drops a message into a named queue, creating it if user requested.
/// \ref      
/// \details Drops a message into a named queue, creating it if user requested.
///   The message is associated a priority still user defined. 
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
///  g++ -std=c++14 -lrt -Wall dropone.cpp -o dropone
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

// name of the POSIX object referencing the queue
const std::string message_queue_object_name {"/myqueue123"};

// max length of a message (just for this process)
constexpr long maximum_message_length {70};

int main()
{
  const unsigned int message_priority {1};

  const pid_t my_pid {::getpid()};
  std::cout << " my_pid : " << my_pid << " \n";

  std::array<char, maximum_message_length> message_content;

  // opening the queue -- ::mq_open()

  // ::mq_open() for creating a new queue (using default attributes)
  MessageQueue message_queue {
    message_queue_object_name.c_str(),
    static_cast<long>(AllFlags::send_and_receive) |
      static_cast<long>(AllFlags::create) |
      static_cast<long>(AllFlags::exclusive_existence),
    static_cast<mode_t>(AllModes::owner_read_write_execute)};

  // producing the message
  std::time_t currtime {std::time(nullptr)};

  std::string hello_message {
    "Hello from process " + std::to_string(my_pid) + " (at " +
      std::ctime(&currtime)};

  std::cout << " hello_message : " << hello_message << '\n';
  std::cout << " hello_message.size() : " << hello_message.size() << '\n';

  // WORKS
  std::size_t length {hello_message.copy(message_content.data(),
    hello_message.size(), 0)};

  std::cout << " length : " << length << '\n';
  std::cout << " message_content.data() " <<
    std::string{message_content.data()} << '\n';

  std::cout << " message_content.size() : " << message_content.size() << '\n';

  // "sending the message" -- ::mq_send()
  message_queue.add_to_queue(
    message_content.data(),
    message_content.size(),
    message_priority);

}

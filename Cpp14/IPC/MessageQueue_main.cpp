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
///  g++ -std=c++14 -lrt MessageQueue_main.cpp -o MessageQueue_main
//------------------------------------------------------------------------------
//#include "MessageQueue/MessageQueue.h"
#include "MessageQueue.h"

#include "../Utilities/casts.h" // get_underlying_value

#include <iostream>
#include <mqueue.h>

//using IPC::AllFlags;
//using IPC::AllModes;
//using IPC::MessageAttributes;
//using IPC::MessageQueue;
//using IPC::maximum_message_size;
//using IPC::maximum_number_of_messages;

using IPC::MQ::AdditionalOperationFlags;
using IPC::MQ::Attributes;
using IPC::MQ::OperationFlags;
using Utilities::get_underlying_value;

int main()
{

  #if 0
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

  // AllModesRepresentsModes
  std::cout << "\n AllModesRepresentsModes \n";  
  std::cout << " AllModes::owner_read_write_execute : " << 
    static_cast<mode_t>(AllModes::owner_read_write_execute) << '\n'; // 0
  std::cout << " AllModes::owner_read : " << 
    static_cast<mode_t>(AllModes::owner_write) << '\n'; // 1
  std::cout << " AllModes::owner_write : " << 
    static_cast<mode_t>(AllModes::owner_write) << '\n'; // 2 
  std::cout << " AllModes::owner_execute : " << 
    static_cast<mode_t>(AllModes::owner_execute) << '\n'; // 2 
  std::cout << (static_cast<mode_t>(AllModes::owner_read) |
    static_cast<mode_t>(AllModes::owner_write)) << '\n';
  std::cout << static_cast<mode_t>(AllModes::all_read_write)  << '\n';


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
  MessageAttributes attributes {};
  std::cout << attributes.mq_flags << ' ' <<
    attributes.mq_maxmsg << ' ' << attributes.mq_msgsize <<
    ' ' << attributes.mq_curmsgs << '\n';

  // MessageAttributesCopyConstructs
  {
    std::cout << "\n MessageAttributesCopiesWith::mq_setattr: \n";

    // maximum number of messages allowed in queue
    constexpr long test_maximum_messages {16};

    constexpr long test_maximum_message_size {sizeof(int)};

    MessageAttributes attributes1 {
      static_cast<long>(AllFlags::send_and_receive) |
        static_cast<long>(AllFlags::create) |
        static_cast<long>(AllFlags::exclusive_existence),
      test_maximum_messages,
      test_maximum_message_size        
    };

    std::cout << attributes1.mq_flags << ' ' <<
      attributes1.mq_maxmsg << ' ' << attributes1.mq_msgsize <<
      ' ' << attributes1.mq_curmsgs << '\n';

    MessageAttributes attributes2 {attributes1};

    std::cout << attributes2.mq_flags << ' ' <<
      attributes2.mq_maxmsg << ' ' << attributes2.mq_msgsize <<
      ' ' << attributes2.mq_curmsgs << '\n';

  }

  // MessageQueueConstructsVia::openWithMessageAttributes
  // \ref https://stackoverflow.com/questions/3056307/how-do-i-use-mqueue-in-a-c-program-on-a-linux-based-system
  {
    std::cout << " MessageQueueConstructsVia::openWithMessageAttributes: \n";

    // maximum number of messages allowed in queue
    constexpr long test_maximum_messages {16};

    const std::string queue_name {"/test_queue"};
    const std::string queue_name1 {"/test_queue1"};

    constexpr long test_maximum_message_size {sizeof(int)};
    std::cout << " test_maximum_message_size: " <<
      test_maximum_message_size << '\n';

    MessageAttributes attributes {
      static_cast<long>(AllFlags::send_and_receive) |
        static_cast<long>(AllFlags::create) |
        static_cast<long>(AllFlags::exclusive_existence),
      test_maximum_messages,
      test_maximum_message_size        
    };

    MessageAttributes attributes1 {
      static_cast<long>(AllFlags::send_and_receive) |
        static_cast<long>(AllFlags::create) |
        static_cast<long>(AllFlags::exclusive_existence),
      test_maximum_messages,
      test_maximum_message_size        
    };

    std::cout << attributes.mq_flags << ' ' <<
      attributes.mq_maxmsg << ' ' << attributes.mq_msgsize <<
      ' ' << attributes.mq_curmsgs << '\n';
#endif 

#if 0
    const int mq_open_result {
      ::mq_open(
        queue_name.c_str(),
        static_cast<long>(AllFlags::send_and_receive) |
          static_cast<long>(AllFlags::create) |
          static_cast<long>(AllFlags::exclusive_existence),
        static_cast<mode_t>(AllModes::owner_read_write_execute),
        attributes.to_mq_attr()
        )};

    std::cout << " mq_open_result : " << mq_open_result << std::endl;

    MessageQueue message_queue1 {
      queue_name1,
      static_cast<long>(AllFlags::send_and_receive) |
        static_cast<long>(AllFlags::create) |
        static_cast<long>(AllFlags::exclusive_existence),
      static_cast<mode_t>(AllModes::owner_read_write_execute),
      attributes1};
  }
#endif 

  // OperationFlags
  {
    std::cout << "\n OperationFlags \n";

    std::cout << " OperationsFlags::receive_only : " <<
      get_underlying_value<OperationFlags>(OperationFlags::receive_only) <<
      '\n'; // 0
    std::cout << " OperationsFlags::send_only : " <<
      get_underlying_value<OperationFlags>(OperationFlags::send_only) << '\n';
        // 1
    std::cout << " OperationsFlags::send_and_receive : " <<
      get_underlying_value<OperationFlags>(OperationFlags::send_and_receive) <<
      '\n'; // 2
    std::cout << " AdditionalOperationsFlags::close_on_execution : " <<
      get_underlying_value<AdditionalOperationFlags>(
        AdditionalOperationFlags::close_on_execution)
        << '\n'; // 524288
    std::cout << " AdditionalOperationsFlags::create : " <<
      get_underlying_value<AdditionalOperationFlags>(
        AdditionalOperationFlags::create) << '\n';
        // 64
    std::cout << " AdditionalOperationsFlags::exclusive_existence : " <<
      get_underlying_value<AdditionalOperationFlags>(
        AdditionalOperationFlags::exclusive_existence)
        << '\n'; // 128
    std::cout << " AdditionalOperationsFlags::nonblocking : " <<
      get_underlying_value<AdditionalOperationFlags>(
        AdditionalOperationFlags::nonblocking) << '\n';
        // 2048
  }

  // AttributesConstructsCorrectly
  {
    std::cout << " \n AttributesConstructsCorrectly \n";

    const Attributes attributes {5, 8};
    std::cout << attributes << '\n';
  }
}


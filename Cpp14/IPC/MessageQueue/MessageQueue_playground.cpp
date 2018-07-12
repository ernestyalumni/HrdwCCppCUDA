// \name MessageQueue_playground.cpp

#include <fcntl.h>
#include <iostream>
#include <mqueue.h>

int main()
{
  {
    // maximum number of messages
    constexpr long maximum_number_of_messages {16};

    constexpr long maximum_message_size {sizeof(int)};
    std::cout << sizeof(int) << '\n';

//    std::cout < " maximum_message_size : " << maximum_message_size << '\n';

    ::mq_attr message_queue_attributes {
      O_CREAT | O_EXCL | O_RDWR,
      maximum_number_of_messages,
      maximum_message_size
    };

  }
}

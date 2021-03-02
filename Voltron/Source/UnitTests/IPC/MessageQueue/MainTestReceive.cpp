//------------------------------------------------------------------------------
/// \file MainTestReceive.cpp
/// \author Ernest Yeung
/// \brief This test goes with MainTestSend.cpp. MainTestSend.cpp does a loop of
/// ::mq_sends, and MainTestReceive.cpp does a loop of mq_receives.
/// \details int main() is needed.
/// \ref https://www.cs.auckland.ac.nz/references/unix/digital/APS33DTE/DOCU_011.HTM#que-ex2
///-----------------------------------------------------------------------------

#include "IPC/MessageQueue/CreateOrOpen.h"
#include "IPC/MessageQueue/FlagsModesAttributes.h"
#include "IPC/MessageQueue/MessageQueueDescription.h"
#include "IPC/MessageQueue/Receive.h"

#include <iostream>
#include <string>

using IPC::MessageQueue::AdditionalOperationFlags;
using IPC::MessageQueue::CreateOrOpen;
using IPC::MessageQueue::MessageQueueDescription;
using IPC::MessageQueue::OpenConfiguration;
using IPC::MessageQueue::OperationFlags;
using IPC::MessageQueue::Receive;

int main()
{
  constexpr mode_t p_mode {0666};

  constexpr std::size_t P4IPC_MSGSIZE {128};

  std::cout << "START OF TEST_RECEIVE \n";

  const std::string queue_name {"/myipc"};

  // Specify O_RDONLY since we only plan to write to the queue, although we
  // could specify O_RDWR also.
  OpenConfiguration open_config {queue_name, OperationFlags::receive_only};
  // Specify O_CREAT so that the file will get created if it doesn't already
  // exist.
  open_config.add_additional_operation(AdditionalOperationFlags::create);

  // Fill in attributes for message queue, via CreateOrOpen.

  CreateOrOpen create_queue {open_config, p_mode, 10, P4IPC_MSGSIZE};

  // Set the flags for the open of the queue.
  // Make it a blocking open on the queue,
  // meaning it will block if this process tries to
  // send to the queue and the queue is full.
  // (Absence of O_NONBLOCK flag implies that
  // the open is blocking)
  //
  // Specify O_CREAT so that the file will get
  // created if it does not already exist.
  //
  // Specify O_RDONLY since we are only
  // planning to write to the queue,
  // although we could specify O_RDWR also.

  // Open the queue, and create it if the sending process hasn't
  // already created it.

  // Open the queue, and create it if the receiving process hasn't already
  // created it.
  const auto create_result = create_queue();

  if (create_result.first)
  {
    std::cout << "Was there a mqd created? " <<
      static_cast<bool>(create_result.second) << "\n";

    std::cerr << create_result.first->error_number() <<
      create_result.first->as_string() << "\n";

    exit(0);
  }

  MessageQueueDescription description {
    *create_result.second,
    queue_name,
    MessageQueueDescription::Finish::CloseAndUnlink};

  Receive receiver {description};

  // Perform the receive 10 times.
  for (int i {0}; i < 10; ++i)
  {
    const auto receive_result = receiver.receive_message<P4IPC_MSGSIZE>();

    if (receive_result.first)
    {
      std::cout << "\n mq_receive failure on mqfd \n";
      std::cerr << receive_result.first->error_number() <<
        receive_result.first->as_string() << "\n";
    }
    else
    {
      std::cout << "Data read for iteration " << i << " = " <<
        receive_result.second.buffer_string() << "\n";
    }

  }

  // Done with queue, so close it.

  // Done with test, so unlink the queue,
  // which destroys it.
  // You only need one call to unlink.

  std::cout << "Exiting receiving process after closing and unlinking queue \n";
}
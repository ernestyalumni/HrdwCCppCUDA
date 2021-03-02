//------------------------------------------------------------------------------
/// \file MainTestSend.cpp
/// \author Ernest Yeung
/// \brief Main file to drope a message into a defined queue, creating it if
/// user requested. The message is associated a priority, possibly user defined.
///
/// \details int main() is needed.
/// \ref https://www.cs.auckland.ac.nz/references/unix/digital/APS33DTE/DOCU_011.HTM
/// https://www.cs.auckland.ac.nz/references/unix/digital/APS33DTE/DOCU_011.HTM#que-ex1
///-----------------------------------------------------------------------------

#include "IPC/MessageQueue/CreateOrOpen.h"
#include "IPC/MessageQueue/FlagsModesAttributes.h"
#include "IPC/MessageQueue/MessageQueueDescription.h"
#include "IPC/MessageQueue/Send.h"

#include <array>
#include <iostream>
// http://www-users.mat.umk.pl/~jb/PS/PS1/cwiczenia/linux/posix1b/mqueue.h
#include <mqueue.h> // P4IPC_MSGSIZE 
// Prints a textual description of error code currently stored in system
// variable errno to stderr.
//#include <stdio.h> // perror
#include <string>
#include <sys/stat.h> // mode_t
#include <unistd.h> // ::get_current_dir_name()

using IPC::MessageQueue::AdditionalOperationFlags;
using IPC::MessageQueue::CreateOrOpen;
using IPC::MessageQueue::MessageQueueDescription;
using IPC::MessageQueue::ModePermissions;
using IPC::MessageQueue::OpenConfiguration;
using IPC::MessageQueue::OperationFlags;
using IPC::MessageQueue::Send;
using IPC::MessageQueue::Unlink;
///*
int main()
{
  constexpr mode_t p_mode {0666};

  constexpr std::size_t P4IPC_MSGSIZE {128};

  std::array<char, P4IPC_MSGSIZE> message_buffer;

  std::cout << "START OF TEST_SEND \n";

  const std::string current_dir_name_str {::get_current_dir_name()};
  //const std::string queue_name {current_dir_name_str + "/myipc"};
  const std::string queue_name {"/myipc"};

  // cf. https://stackoverflow.com/questions/25799895/cannot-set-posix-message-queue-attribute
  //const auto unlink_result = Unlink()(queue_name);
  //std::cout << "No queue existed before: " << static_cast<bool>(unlink_result)
  //  << "\n";

  // Set the flags for the open of the queue.
  // Make it a blocking open on the queue, meaning it will block if this process
  // tries to send to the queue and queue is full.
  // (Absence of O_NONBLOCK flag implies open is blocking)
  //
  // Specify O_WRONLY since we only plan to write to the queue, although we
  // could specify O_RDWR also.
  OpenConfiguration open_config {queue_name, OperationFlags::send_only};
  // Specify O_CREAT so that the file will get created if it doesn't already
  // exist.
  open_config.add_additional_operation(AdditionalOperationFlags::create);

  // \ref http://www-users.mat.umk.pl/~jb/PS/PS1/cwiczenia/linux/posix1b/mqueue.h
  // P4IPC_MSGSIZE - Implementation specified default size in bytes of a
  // message. Used when attr is not specified by the user in mq_open().

  // number of messages highly dependent on message queue set on OS.
  CreateOrOpen create_queue {open_config, p_mode, 10, P4IPC_MSGSIZE};
  /*
  CreateOrOpen create_queue {
    open_config,
    ModePermissions::user_rwx,
    20,
    P4IPC_MSGSIZE
  };*/
///*
  create_queue.add_additional_permissions(ModePermissions::group_rwx);

  std::cout << "\n Has Attributes? " <<
    static_cast<bool>(create_queue.attributes()) << "\n";

  if (create_queue.attributes())
  {
    std::cout << "\n flags : " << create_queue.attributes()->flags() << "\n";
    std::cout << "\n maximum_number_of_messages : " <<
      create_queue.attributes()->maximum_number_of_messages() << "\n";
    std::cout << "\n maximum_message_size : " <<
      create_queue.attributes()->maximum_message_size() << "\n";
    std::cout << "\n current number of messages in queue : " <<
      create_queue.attributes()->current_number_of_messages_in_queue() << "\n";
  }

  std::cout << "\n queue_name in configuration: " <<
    create_queue.configuration().name() << "\n";

  std::cout << "\n operation_flag in configuration: " <<
    create_queue.configuration().operation_flag() << "\n";

  std::cout << "\n O_WRONLY|O_CREAT : " << (O_WRONLY|O_CREAT) << "\n";

  std::cout << "\n mode permission in create_queue: " <<
    create_queue.mode() << "\n";

  std::cout << "S_IRWXU | S_IRWXG :" << (S_IRWXU | S_IRWXG) << "\n";

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
    queue_name};
//    MessageQueueDescription::Finish::CloseAndUnlink};

  // Uncomment this out for the actual, closes only, MessageQueueDescription
  // that'll be used in production code.
  //MessageQueueDescription description {*create_result.second, queue_name};

  // Fill in a test message buffer to send.
  message_buffer = {'P','R','I','O','R','I','T','Y','1','a'};

  Send sender {description};

  constexpr int number_of_bytes_to_send {10};
  constexpr int priority_of_message {1};

  // Perform the send 10 times.
  for (int i {0}; i < 10; ++i)
  {
    const auto send_result =
      sender(message_buffer, number_of_bytes_to_send, priority_of_message);

    if (send_result)
    {
      std::cerr << send_result->as_string() << "\n";
    }
    else
    {
      std::cout << "Successful call to ::mq_send, i = " << i << "\n";
    }
  }

  std::cout << "About to exit the sending process after closing the queue \n";

  return 0;
}
//*/ 

/*
 * test_send.c
 *
 * This test goes with test_receive.c.
 * test_send.c does a loop of mq_sends,
 * and test_receive.c does a loop of mq_receives.
 */
/*
#include <unistd.h>
#include <sys/types.h>
#include <stdio.h>
#include <sys/time.h>
#include <sys/resource.h>
#include <time.h>
#include <sched.h>
#include <sys/mman.h>
#include <sys/fcntl.h>
#include <signal.h>
//#include <sys/rt_syscall.h>
#include <mqueue.h>
#include <errno.h>

#include <iostream>

#define PMODE 0666
extern int errno;

int main()
{

  constexpr std::size_t P4IPC_MSGSIZE {128};
//  constexpr std::size_t P4IPC_MSGSIZE {100};


int i;
int status = 0;
mqd_t mqfd;
char msg_buffer[P4IPC_MSGSIZE];
struct mq_attr attr;
int open_flags = 0;
int num_bytes_to_send;
int priority_of_msg;

printf("START OF TEST_SEND \n");

// */
/* Fill in attributes for message queue */
/*
// https://stackoverflow.com/questions/29382550/why-is-errno-set-to-22-mq-open-posix
// EY: 20200611, for mq_maxmsg = 20, it fails.
attr.mq_maxmsg = 10;
attr.mq_msgsize = static_cast<long>(P4IPC_MSGSIZE);
attr.mq_flags   = 0;

attr.mq_curmsgs = 0;

std::cout << "\n attr.mq_flags : " << attr.mq_flags << "\n";
std::cout << "\n attr.mq_curmsgs : " << attr.mq_curmsgs << "\n";

/* Set the flags for the open of the queue.
 * Make it a blocking open on the queue, meaning it will block if
 * this process tries to send to the queue and the queue is full.
 * (Absence of O_NONBLOCK flag implies that the open is blocking)
 *
 * Specify O_CREAT so that the file will get created if it does not
 * already exist.
 *
 * Specify O_WRONLY since we are only planning to write to the queue,
 * although we could specify O_RDWR also.
 */
//open_flags = O_WRONLY|O_CREAT;

/* Open the queue, and create it if the receiving process hasn't
 * already created it.
 */
/*
//https://stackoverflow.com/questions/25799895/cannot-set-posix-message-queue-attribute
int mq_unlink_result {::mq_unlink("/myipc")};
std::cout << "\n mq_unlink_result :" << mq_unlink_result << "\n";

mqfd = mq_open("/myipc",open_flags,PMODE,&attr);
//mqfd = mq_open("/myipc",open_flags,PMODE,nullptr);
if (mqfd == -1)
    {
    perror("mq_open failure from main");
    exit(0);
    };
//*/
/* Fill in a test message buffer to send */
/*
msg_buffer[0] = 'P';
msg_buffer[1] = 'R';
msg_buffer[2] = 'I';
msg_buffer[3] = 'O';
msg_buffer[4] = 'R';
msg_buffer[5] = 'I';
msg_buffer[6] = 'T';
msg_buffer[7] = 'Y';
msg_buffer[8] = '1';
msg_buffer[9] = 'a';

num_bytes_to_send = 10;
priority_of_msg = 1;

/* Perform the send 10 times */
/*for (i=0; i<10; i++)
    {
    status = mq_send(mqfd,msg_buffer,num_bytes_to_send,priority_of_msg);
    if (status == -1)
        perror("mq_send failure on mqfd");
    else
        printf("successful call to mq_send, i = %d\n",i);
    }
//*/
/* Done with queue, so close it */
/*
if (mq_close(mqfd) == -1)
    perror("mq_close failure on mqfd");

printf("About to exit the sending process after closing the queue \n");

::mq_unlink("/myipc");
}
//*/
//------------------------------------------------------------------------------
/// \file MessageQueue.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  POSIX message queue as RAII (Resource Acquisition Is
/// Initialization).
/// \ref http://man7.org/linux/man-pages/man3/mq_open.3.html   
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
///
/// \details
//------------------------------------------------------------------------------
/// COMPILATION TIPS:
///  g++ -std=c++14 MessageQueue_main.cpp -o MessageQueue_main
//------------------------------------------------------------------------------
#ifndef _IPC_MESSAGE_QUEUE_H_
#define _IPC_MESSAGE_QUEUE_H_

#include "../Utilities/CheckReturn.h" // CheckReturn, check_close
#include "../Utilities/casts.h" // get_underlying_value

#include <algorithm> // std::find_if
#include <fcntl.h> // For O_* constants
#include <iostream>
#include <mqueue.h> // ::mq_attr
#include <stdexcept> // std::out_of_range
#include <string> // std::to_string
#include <type_traits> 
#include <unistd.h> // ::close
#include <utility> // std::pair, std::get
#include <vector>

namespace IPC
{

// Message Queue (MQ)
namespace MQ
{

//------------------------------------------------------------------------------
/// \brief enum class for all flags that control operation of the call
/// ::mq_open().
/// \ref https://manpages.debian.org/testing/manpages-dev/mq_open.3.en.html
//------------------------------------------------------------------------------
enum class OperationFlags : int
{
  receive_only = O_RDONLY, // Open the queue to receive messages only.
  send_only = O_WRONLY, // Open the queue to send messages only.
  send_and_receive = O_RDWR, // Open queue to both send and receive messages.
};

//------------------------------------------------------------------------------
/// \brief enum class for all additional and optional flags that control
/// operation of the call ::mq_open().
/// \ref https://manpages.debian.org/testing/manpages-dev/mq_open.3.en.html
//------------------------------------------------------------------------------
enum class AdditionalOperationFlags : int
{
  close_on_execution = O_CLOEXEC,
  create = O_CREAT, // Create message queue if it doesn't exist
  exclusive_existence = O_EXCL, // Fail if queue with given name already exists
  nonblocking = O_NONBLOCK // non-blocking mode  
};

//------------------------------------------------------------------------------
/// \brief Derived class for ::mq_attr
/// \ref http://man7.org/linux/man-pages/man3/mq_open.3.html
/// \details Fields of struct mq_attr specify maximum number of messages and
/// the maximum size of messages that the queue will allow, defined as
/// 
/// struct mq_attr
/// {
///   long mq_flags; // Flags (ignored for mq_open())
///   long mq_maxmsg; // Max. # of messages on queue 
///   long mq_msgsize; // Max. message size (bytes)
///   long mq_curmsgs; // # of messages currently in queue (ignored for
///   // mq_open())
/// };
///
//------------------------------------------------------------------------------
struct Attributes : public ::mq_attr
{
  Attributes(const long max_msg, const long max_msg_size)
  {
    this->mq_maxmsg = max_msg;
    this->mq_msgsize = max_msg_size;
  }

  Attributes(
    const long flags,
    const long max_msg,
    const long max_msg_size,
    const long current_msgs):
    ::mq_attr{flags, max_msg, max_msg_size, current_msgs}
  {}

  Attributes(const long flags, const long max_msg, const long max_msg_size):
    ::mq_attr{flags, max_msg, max_msg_size}
  {}

  Attributes() = default;

  // Consider using &epoll_event for Epollevent epoll_event, instead.
  const ::mq_attr* to_mq_attr() const
  {
    return reinterpret_cast<const ::mq_attr*>(this);
  }

  ::mq_attr* to_mq_attr()
  {
    return reinterpret_cast<::mq_attr*>(this);
  }

  friend std::ostream& operator<<(
    std::ostream& os,
    const Attributes& attributes);
};

std::ostream& operator<<(std::ostream& os, const Attributes& attributes)
{
  os << attributes.mq_flags << ' ' << attributes.mq_maxmsg << ' ' <<
    attributes.mq_msgsize << ' ' << attributes.mq_curmsgs << '\n';

  return os;
}

class MessageQueue
{
  public:

  private:

    class CheckMessageQueueOpen;
};

} // namespace MQ

} // namespace IPC

#endif // _IPC_MESSAGE_QUEUE_H_

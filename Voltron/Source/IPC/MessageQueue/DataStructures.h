//------------------------------------------------------------------------------
/// \file DataStructures.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref http://man7.org/linux/man-pages/man2/mq_open.2.html
/// \brief Wrapper for message queue flags and attributes.
//------------------------------------------------------------------------------
#ifndef IPC_MESSAGE_QUEUE_DATA_STRUCTURES_H
#define IPC_MESSAGE_QUEUE_DATA_STRUCTURES_H

#include <fcntl.h> // For O_* constants
#include <mqueue.h> // ::mq_attr
#include <ostream>
#include <sys/stat.h> // permission bits, mode_t

namespace IPC
{
namespace MessageQueue
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

// mode_t is the mode of the file
// cf. https://pubs.opengroup.org/onlinepubs/007908775/xsh/sysstat.h.html
enum class ModePermissions : mode_t
{
	user_rwx = S_IRWXU, // 00700 user (file owner) has read, write, and execute
	// permission
	user_read = S_IRUSR, // 00400 user has read permission
	user_write = S_IWUSR, // 00200 user has write permission
	user_exec = S_IXUSR, // 00100 user has execute permission
	group_rwx = S_IRWXG, // 00070 group has read, write and execute permission
	group_read = S_IRGRP, // 00040 group has read permission
  group_write = S_IWGRP, // 00020 group has write permission
  group_exec = S_IXGRP, // 00010 group has execute permission
  others_rwx = S_IRWXO, // 00007 others have read, write, and execute permission
  others_read = S_IROTH // 00004 others have read permission
	// TODO: fill up rest.
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

  // friend function is not a member function, so define it here; cannot define
  // it as member function outside.
  friend std::ostream& operator<<(
    std::ostream& os,
    const Attributes& attributes)
  {
    os << attributes.mq_flags << ' ' << attributes.mq_maxmsg << ' ' <<
    attributes.mq_msgsize << ' ' << attributes.mq_curmsgs << '\n';

    return os;
  }
};

} // namespace MessageQueue
} // namespace IPC

#endif // IPC_MESSAGE_QUEUE_DATA_STRUCTURES_H

//------------------------------------------------------------------------------
/// \file FLAGS_AND_MODES.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref http://man7.org/linux/man-pages/man2/mq_open.2.html
/// \brief Wrapper for message queue flags and MODES.
//------------------------------------------------------------------------------
#ifndef IPC_MESSAGE_QUEUE_FLAGS_AND_MODES_H
#define IPC_MESSAGE_QUEUE_FLAGS_AND_MODES_H

#include <fcntl.h> // For O_* constants
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
enum class OperationFlag : int
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
enum class AdditionalOperationFlag : int
{
  close_on_execution = O_CLOEXEC,
  create = O_CREAT, // Create message queue if it doesn't exist
  exclusive_existence = O_EXCL, // Fail if queue with given name already exists
  nonblocking = O_NONBLOCK // non-blocking mode  
};

int to_operation_flag_value(const OperationFlag operation_flag);

OperationFlag to_operation_flag(const int value)
{
  return static_cast<OperationFlag>(value);
}

int to_additional_operation_flag_value(const AdditionalOperationFlag flag);

AdditionalOperationFlag to_additional_operation_flag(const int value)
{
  return static_cast<AdditionalOperationFlag>(value);
}

//------------------------------------------------------------------------------
/// \brief Specifies permissions to be placed on a new queue.
/// \details mode_t is the mode of the file
/// \ref https://pubs.opengroup.org/onlinepubs/007908775/xsh/sysstat.h.html
/// https://man7.org/linux/man-pages/man2/open.2.html
//------------------------------------------------------------------------------
enum class PermissionMode : mode_t
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
  others_read = S_IROTH, // 00004 others have read permission
  others_write = S_IWOTH, // 00002 others have write permission
  others_exec = S_IXOTH, // 00001 others have execute permission
  set_user_id = S_ISUID, // set-user-ID on execution, 04000
  set_group_id = S_ISGID, // set group ID on execution, 02000
  on_directories_restricted_deletion = S_ISVTX, // on directories, restricted
  // deletion flag. 01000

  // User defined.
  read_write_all = 0666  
};

mode_t to_permission_mode_value(const PermissionMode mode);

PermissionMode to_permission_mode(const mode_t value)
{
  return static_cast<PermissionMode>(value);
}

} // namespace MessageQueue
} // namespace IPC

#endif // IPC_MESSAGE_QUEUE_FLAGS_AND_MODES_H

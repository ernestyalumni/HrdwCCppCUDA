//------------------------------------------------------------------------------
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref https://man7.org/linux/man-pages/man3/mq_open.3.html
//------------------------------------------------------------------------------
#ifndef IPC_MESSAGE_QUEUE_CREATE_MESSAGE_QUEUE_H
#define IPC_MESSAGE_QUEUE_CREATE_MESSAGE_QUEUE_H

#include "FlagsAndModes.h"
#include "Utilities/ErrorHandling/HandleError.h"

#include <cstddef> // std::size_t
#include <set>
#include <string>
#include <sys/stat.h> // mode_t

namespace IPC
{
namespace MessageQueue
{

class CreateMessageQueue
{
  public:

    //--------------------------------------------------------------------------
    /// \ref http://man7.org/linux/man-pages/man2/mq_open.2.html
    /// \details mqd_t mq_open(const char* name, int oflag, mode_t mode,
    /// struct mq_attr* attr)
    /// oflag is exactly 1 of the following, O_RDONLY, O_WRONLY, O_RDWR
    /// and 0 or more additional flags.
    /// mode specifies permissions to be placed on new queue, as for ::open 
    //--------------------------------------------------------------------------
    CreateMessageQueue(
      const std::string& queue_name,
      const OperationFlag operation_flag,
      const mode_t mode,
      const std::size_t maximum_number_of_messages,
      const std::size_t maximum_message_size);

    CreateMessageQueue(
      const std::string& queue_name,
      const OperationFlag operation_flag,
      const PermissionMode mode,
      const std::size_t maximum_number_of_messages,
      const std::size_t maximum_message_size);

    CreateMessageQueue(
      const std::string& queue_name,
      const OperationFlag operation_flag,
      const std::size_t maximum_number_of_messages,
      const std::size_t maximum_message_size);

    CreateMessageQueue(
      const std::string& queue_name,
      const OperationFlag operation_flag,
      const mode_t mode);

    CreateMessageQueue(
      const std::string& queue_name,
      const OperationFlag operation_flag,
      const PermissionMode mode);

    CreateMessageQueue(
      const std::string& queue_name,
      const OperationFlag operation_flag);

    void add_additional_operation_flag(const AdditionalOperationFlag flag);

    void add_additional_permission_mode(const mode_t mode);

    void add_additional_permission_mode(const PermissionMode mode);

    class HandleCreateError : public Utilities::ErrorHandling::HandleError
    {
      public:
        
        HandleCreateError();

        virtual void operator()(const int result);
    };

  private:

    std::string queue_name_;
    int operation_flag_value_;
    OperationFlag operation_flag_;
    mode_t mode_value_;
    std::set<AdditionalOperationFlag> additional_operation_flags_;
    long maximum_number_of_messages_;
    long maximum_message_size_;
};

} // namespace MessageQueue
} // namespace IPC

#endif // IPC_MESSAGE_QUEUE_CREATE_MESSAGE_QUEUE_H

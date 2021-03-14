#include "CreateMessageQueue.h"

#include "FlagsAndModes.h"

namespace IPC
{
namespace MessageQueue
{

CreateMessageQueue::CreateMessageQueue(
  const std::string& queue_name,
  const OperationFlag operation_flag,
  const mode_t mode,
  const std::size_t maximum_number_of_messages,
  const std::size_t maximum_message_size
  ):
  queue_name_{queue_name},
  operation_flag_value_{to_operation_flag_value(operation_flag)},
  operation_flag_{operation_flag},
  mode_value_{mode},
  additional_operation_flags_{},
  maximum_number_of_messages_{static_cast<long>(maximum_number_of_messages)},
  maximum_message_size_{static_cast<long>(maximum_message_size)}
{
  add_additional_operation_flag(AdditionalOperationFlag::create);
}

CreateMessageQueue::CreateMessageQueue(
  const std::string& queue_name,
  const OperationFlag operation_flag,
  const PermissionMode mode,
  const std::size_t maximum_number_of_messages,
  const std::size_t maximum_message_size
  ):
  CreateMessageQueue{
    queue_name,
    operation_flag,
    to_permission_mode_value(mode),
    maximum_number_of_messages,
    maximum_message_size}
{
  // It should execute ctor body of delegated constructor from above.
}

CreateMessageQueue::CreateMessageQueue(
  const std::string& queue_name,
  const OperationFlag operation_flag,
  const std::size_t maximum_number_of_messages,
  const std::size_t maximum_message_size
  ):
  CreateMessageQueue{
    queue_name,
    operation_flag,
    to_permission_mode_value(PermissionMode::read_write_all),
    maximum_number_of_messages,
    maximum_message_size}
{
  // It should execute ctor body of delegated constructor from above.
}

CreateMessageQueue::CreateMessageQueue(
  const std::string& queue_name,
  const OperationFlag operation_flag,
  const mode_t mode
  ):
  queue_name_{queue_name},
  operation_flag_value_{to_operation_flag_value(operation_flag)},
  operation_flag_{operation_flag},
  mode_value_{mode},
  additional_operation_flags_{},
  maximum_number_of_messages_{-1},
  maximum_message_size_{-1}
{
  add_additional_operation_flag(AdditionalOperationFlag::create);
}

CreateMessageQueue::CreateMessageQueue(
  const std::string& queue_name,
  const OperationFlag operation_flag,
  const PermissionMode mode
  ):
  CreateMessageQueue{
    queue_name,
    operation_flag,
    to_permission_mode_value(mode)}
{
  // It should execute ctor body of delegated constructor from above.  
}

CreateMessageQueue::CreateMessageQueue(
  const std::string& queue_name,
  const OperationFlag operation_flag
  ):
  CreateMessageQueue{
    queue_name,
    operation_flag,
    to_permission_mode_value(PermissionMode::read_write_all)}
{
  // It should execute ctor body of delegated constructor from above.  
}

void CreateMessageQueue::add_additional_operation_flag(
  const AdditionalOperationFlag flag)
{
  additional_operation_flags_.insert(flag);

  operation_flag_value_ |= to_additional_operation_flag_value(flag);
}

void CreateMessageQueue::add_additional_permission_mode(const mode_t mode)
{
  mode_value_ |= mode;
}

void CreateMessageQueue::add_additional_permission_mode(
  const PermissionMode mode)
{
  add_additional_permission_mode(to_permission_mode_value(mode));
}

} // namespace MessageQueue
} // namespace IPC

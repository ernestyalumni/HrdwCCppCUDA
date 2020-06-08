//------------------------------------------------------------------------------
/// \file CreateOrOpen.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref http://man7.org/linux/man-pages/man2/mq_open.2.html
/// \brief Wrapper for ::mq_open, which creates a new POSIX message queue or
/// opens an existing queue.
//------------------------------------------------------------------------------
#include "CreateOrOpen.h"

#include "IPC/MessageQueue/DataStructures.h"

#include <mqueue.h> // ::mqd_t
#include <optional>
#include <string>
#include <sys/stat.h> // mode_t
#include <utility>

using Utilities::ErrorHandling::ErrorNumber;

namespace IPC
{
namespace MessageQueue
{

CreateOrOpen::CreateOrOpen(const std::string& name, const int operation_flag):
  name_{name},
  operation_flag_{operation_flag},
  new_queue_inputs_{std::nullopt}
{}

/*
CreateOrOpen::NewQueueInputs CreateOrOpen::fill_in_new_queue_inputs(
  const mode_t mode,
  const long maximum_number_of_messages,
  const long maximum_message_size)
{
  return NewQueueInputs{mode, maximum_number_of_messages, maximum_message_size};
}
*/

CreateOrOpen::CreateOrOpen(
  const std::string& name,
  const int operation_flag,
  const mode_t mode,
  const long maximum_number_of_messages,
  const long maximum_message_size
  ):
  name_{name},
  operation_flag_{operation_flag},
  new_queue_inputs_{
    NewQueueInputs{mode, maximum_number_of_messages, maximum_message_size}}
{}

CreateOrOpen::CreateOrOpen(
  const std::string& name,
  const int operation_flag,
  const mode_t mode
  ):
  name_{name},
  operation_flag_{operation_flag},
  new_queue_inputs_{NewQueueInputs{mode}}
{}

std::pair<
  CreateOrOpen::OptionalErrorNumber,
  std::optional<mqd_t>
  > CreateOrOpen::operator()(const bool create_with_default_attributes)
{
  mqd_t return_value;

  if (!new_queue_inputs_)
  {
    return_value = ::mq_open(name_.c_str(), operation_flag_);   
  }
  else if (create_with_default_attributes)
  {
    return_value =
      ::mq_open(
        name_.c_str(),
        operation_flag_,
        new_queue_inputs_->mode_,
        nullptr);    
  }
  else
  {  
    Attributes attributes {
      new_queue_inputs_->maximum_number_of_messages_,
      new_queue_inputs_->maximum_message_size_};

    return_value =
      ::mq_open(
        name_.c_str(),
        operation_flag_,
        new_queue_inputs_->mode_,
        attributes.to_mq_attr());
  }

  OptionalErrorNumber error_number {HandleMqOpen()(return_value)};

  if (error_number)
  {
    return std::make_pair<
      OptionalErrorNumber,
      std::optional<mqd_t>
      >(std::move(error_number), std::nullopt);
  }
  else
  {
    return std::make_pair<
      OptionalErrorNumber,
      std::optional<mqd_t>
      >(std::nullopt, std::make_optional<mqd_t>(return_value));
  }
}

CreateOrOpen::OptionalErrorNumber CreateOrOpen::HandleMqOpen::operator()(
  const mqd_t return_value)
{
  if (return_value < 0)
  {
    get_error_number();

    return std::make_optional<ErrorNumber>(std::move(error_number()));
  }
  else
  {
    return std::nullopt;
  }
}

} // namespace MessageQueue
} // namespace IPC
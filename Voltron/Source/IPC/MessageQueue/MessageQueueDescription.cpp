//------------------------------------------------------------------------------
/// \file MessageQueueDescription.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref http://man7.org/linux/man-pages/man2/mq_open.2.html
/// \brief Wrapper for ::mq_open, which creates a new POSIX message queue or
/// opens an existing queue.
//------------------------------------------------------------------------------
#include "MessageQueueDescription.h"

#include <mqueue.h>
#include <string>

namespace IPC
{
namespace MessageQueue
{

MessageQueueDescription::MessageQueueDescription(
  const mqd_t message_queue_descriptor,
  const std::string& name
  ):
  finishing_method_{Finish::CloseOnly},
  message_queue_descriptor_{message_queue_descriptor},
  name_{name}
{}

MessageQueueDescription::MessageQueueDescription(
  const mqd_t message_queue_descriptor,
  const std::string& name,
  const Finish finishing_method
  ):
  finishing_method_{finishing_method},
  message_queue_descriptor_{message_queue_descriptor},
  name_{name}
{}

MessageQueueDescription::~MessageQueueDescription()
{
  if (finishing_method_ == Finish::CloseOnly)
  {
    Close()(message_queue_descriptor_);
  }
  else if (finishing_method_ == Finish::CloseAndUnlink)
  {
    Close()(message_queue_descriptor_);
    Unlink()(name_.c_str());
  }
}

Close::OptionalErrorNumber Close::operator()(
  const mqd_t message_queue_descriptor)
{
  const int return_value {::mq_close(message_queue_descriptor)};
  return HandleMqClose()(return_value);
}

Unlink::OptionalErrorNumber Unlink::operator()(const std::string& queue_name)
{
  const int return_value {::mq_unlink(queue_name.c_str())};
  return HandleMqUnlink()(return_value);
}

} // namespace MessageQueue
} // namespace IPC
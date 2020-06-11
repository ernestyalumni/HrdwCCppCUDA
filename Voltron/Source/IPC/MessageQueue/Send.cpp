//------------------------------------------------------------------------------
/// \file Send.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref http://man7.org/linux/man-pages/man2/mq_open.2.html
/// \brief Wrapper for ::mq_open, which creates a new POSIX message queue or
/// opens an existing queue.
//------------------------------------------------------------------------------
#include "Send.h"

#include "MessageQueueDescription.h"
#include "Utilities/ErrorHandling/ErrorHandling.h"

#include <cstddef> // std::size_t
#include <mqueue.h> // ::mq_send

using Utilities::ErrorHandling::HandleReturnValuePassively;

namespace IPC
{
namespace MessageQueue
{

Send::Send(const MessageQueueDescription& description):
  message_queue_descriptor_{description.message_queue_descriptor()}
{}

template<std::size_t N>
Send::OptionalErrorNumber Send::operator()(
  const std::array<char, N>& message,
  const unsigned int priority)
{
  return operator()(message.data(), message.size(), priority);
}

/*
template<std::size_t N>
Send::OptionalErrorNumber Send::operator()(
  const std::array<char, N>& message,
  const std::size_t bytes_to_send,
  const unsigned int priority)
{
  return operator()(message.data(), bytes_to_send, priority);
}
*/

Send::OptionalErrorNumber Send::operator()(
  const char* message_ptr,
  const size_t message_length,
  const unsigned int priority)
{
  const int return_value {
    ::mq_send(message_queue_descriptor_,
    message_ptr,
    message_length,
    priority)};

  return HandleReturnValuePassively()(return_value);
}

} // namespace MessageQueue
} // namespace IPC
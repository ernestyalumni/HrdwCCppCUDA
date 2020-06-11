//------------------------------------------------------------------------------
/// \file Receive.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref http://man7.org/linux/man-pages/man2/mq_open.2.html
/// \brief Wrapper for ::mq_open, which creates a new POSIX message queue or
/// opens an existing queue.
//------------------------------------------------------------------------------
#include "Receive.h"

#include "MessageQueueDescription.h"
#include "Utilities/ErrorHandling/ErrorHandling.h"

#include <cstddef>
#include <mqueue.h> // ::mq_send
#include <optional>

using Utilities::ErrorHandling::ErrorNumber;
using Utilities::ErrorHandling::HandleReturnValuePassively;

namespace IPC
{
namespace MessageQueue
{

Receive::Receive(const MessageQueueDescription& description):
  message_queue_descriptor_{description.message_queue_descriptor()}
{}


template<std::size_t N>
std::pair<Receive::OptionalErrorNumber, Receive::ReceivedMessage<N>>
  Receive::receive_message()
{
  ReceivedMessage<N> received_message;

  const ssize_t return_value {
    ::mq_receive(
      message_queue_descriptor_,
      received_message.buffer(),
      N,
      received_message.priority_reference())};

  const OptionalErrorNumber error_number {HandleMqReceive()(return_value)};

  if (error_number)
  {
    return std::make_pair<OptionalErrorNumber, ReceivedMessage<N>>(
      std::move(error_number),
      std::nullopt);
  }
  else
  {
    return std::make_pair<OptionalErrorNumber, ReceivedMessage<N>>(
      std::nullopt,
      std::move(received_message));
  }
}

template<std::size_t N>
Receive::OptionalErrorNumber Receive::receive_message(
  ReceivedMessage<N>& ready_message)
{
  const ssize_t return_value {
    ::mq_receive(
      message_queue_descriptor_,
      ready_message.buffer(),
      N,
      ready_message.priority_reference())};

  return HandleMqReceive()(return_value);
}

Receive::OptionalErrorNumber Receive::HandleMqReceive::operator()(
  const ssize_t return_value)
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
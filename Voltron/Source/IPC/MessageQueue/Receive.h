//------------------------------------------------------------------------------
/// \file Receive.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref https://man7.org/linux/man-pages/man3/mq_send.3.html
/// \brief Wrappers for message queue send
//------------------------------------------------------------------------------
#ifndef IPC_MESSAGE_QUEUE_SEND
#define IPC_MESSAGE_QUEUE_SEND

#include "MessageQueueDescription.h"
#include "Utilities/ErrorHandling/ErrorHandling.h"

#include <array>
#include <mqueue.h>

namespace IPC
{
namespace MessageQueue
{

class Receive
{
  public:

    using OptionalErrorNumber =
      Utilities::ErrorHandling::HandleReturnValuePassively::OptionalErrorNumber;

    Receive(const MessageQueueDescription& description);

    template <std::size_t N>
    OptionalErrorNumber operator()(
      const std::array<N>& message,
      const unsigned int priority);

    OptionalErrorNumber operator()(
      const char* message_ptr,
      const size_t message_length,
      const unsigned int priority);

  private:

    mqd_t message_queue_descriptor_;
};

} // namespace MessageQueue
} // namespace IPC

#endif // IPC_MESSAGE_QUEUE_SEND
//------------------------------------------------------------------------------
/// \file Send.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref https://man7.org/linux/man-pages/man3/mq_send.3.html
/// \brief Wrappers for message queue send.
//------------------------------------------------------------------------------
#ifndef IPC_MESSAGE_QUEUE_SEND
#define IPC_MESSAGE_QUEUE_SEND

#include "MessageQueueDescription.h"
#include "Utilities/ErrorHandling/ErrorHandling.h"

#include <cstddef> // std::size_t
#include <array>
#include <mqueue.h>

namespace IPC
{
namespace MessageQueue
{

class Send
{
  public:

    using OptionalErrorNumber =
      Utilities::ErrorHandling::HandleReturnValuePassively::OptionalErrorNumber;

    Send(const MessageQueueDescription& description);

    template <std::size_t N>
    OptionalErrorNumber operator()(
      const std::array<char, N>& message,
      const unsigned int priority);

    template <std::size_t N>
    OptionalErrorNumber operator()(
      const std::array<char, N>& message,
      const std::size_t bytes_to_send,
      const unsigned int priority)
    {
      return operator()(message.data(), bytes_to_send, priority);
    }

    //--------------------------------------------------------------------------
    /// \ref https://www.man7.org/linux/man-pages/man3/mq_send.3.html
    /// \details int mq_send(mqd_t mqdes, const char *msg_ptr,
    /// size_t msg_len, unsigned int msg_prio);
    /// msg_len specifies length of the message pointed to by msg_ptr;
    /// this length must be less than or equal to queue's mq_msgsize attribute.
    /// msg_prio argument is a nonnegative integer that specifies priority of
    /// this message.
    //--------------------------------------------------------------------------

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
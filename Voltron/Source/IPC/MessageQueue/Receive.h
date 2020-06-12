//------------------------------------------------------------------------------
/// \file Receive.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref https://man7.org/linux/man-pages/man3/mq_receive.3.html
/// \brief Wrappers for message queue receive.
//------------------------------------------------------------------------------
#ifndef IPC_MESSAGE_QUEUE_RECEIVE
#define IPC_MESSAGE_QUEUE_RECEIVE

#include "MessageQueueDescription.h"
#include "Utilities/ErrorHandling/ErrorHandling.h"

#include <array>
#include <cstddef> // std::size_t
#include <mqueue.h>
#include <string>
#include <utility>

namespace IPC
{
namespace MessageQueue
{

class Receive
{
  public:

    using OptionalErrorNumber =
      Utilities::ErrorHandling::HandleReturnValuePassively::OptionalErrorNumber;

    //--------------------------------------------------------------------------
    /// \ref https://man7.org/linux/man-pages/man3/mq_receive.3.html
    /// \details ssize_t mq_receive(mqd_t mqdes, char *msg_ptr, size_t msg_len,
    ///   unsigned int* msg_prio);
    /// mq_receive(...) removes oldest message with highest priority from queue
    /// (FIFO) referred to by message queue descriptor mqdes, and places it in
    /// buffer pointed to by msg_ptr.
    /// msg_len argument specifies size of buffer pointed to by msg_ptr; this
    /// must be greater than or equal to mq_msgsize attribute of queue (see
    /// mq_getattr).
    /// If msg_prio is not NULL, then buffer to which it points is used to
    /// return priority associated with received message.
    //--------------------------------------------------------------------------
    template <std::size_t N>
    class ReceivedMessage
    {
      public:

        ReceivedMessage():
          buffer_{},
          priority_{},
          bytes_received_{}
        {}

        std::array<char, N> buffer() const
        {
          return buffer_;
        }

        char* buffer_data()
        {
          return buffer_.data();
        }

        std::string buffer_string() const
        {
          return std::string{std::begin(buffer_), std::end(buffer_)};
        }

        unsigned int& priority_reference()
        {
          return priority_;
        }

        void bytes_received(const ssize_t bytes_received)
        {
          bytes_received_ = bytes_received;
        }

      private:

        std::array<char, N> buffer_;        
        unsigned int priority_;

        ssize_t bytes_received_;
    };

    Receive(const MessageQueueDescription& description);

    template <std::size_t N>
    std::pair<OptionalErrorNumber, ReceivedMessage<N>> receive_message()
    {
      ReceivedMessage<N> received_message;

      const ssize_t return_value {
        ::mq_receive(
          message_queue_descriptor_,
          received_message.buffer_data(),
          N,
          &received_message.priority_reference())};

      OptionalErrorNumber error_number {HandleMqReceive()(return_value)};

      if (error_number)
      {
        return std::make_pair<OptionalErrorNumber, ReceivedMessage<N>>(
          std::move(error_number),
          std::move(received_message));
      }
      else
      {
        return std::make_pair<OptionalErrorNumber, ReceivedMessage<N>>(
          std::nullopt,
          std::move(received_message));
      }
    }

    template <std::size_t N>
    OptionalErrorNumber receive_message(ReceivedMessage<N>& ready_message)
    {
      const ssize_t return_value {
        ::mq_receive(
          message_queue_descriptor_,
          ready_message.buffer_data(),
          N,
          &ready_message.priority_reference())};

      return HandleMqReceive()(return_value);      
    }

  private:

    class HandleMqReceive :
      public Utilities::ErrorHandling::HandleReturnValuePassively
    {
      public:

        OptionalErrorNumber operator()(const ssize_t return_value);
    };

    mqd_t message_queue_descriptor_;
};

} // namespace MessageQueue
} // namespace IPC

#endif // IPC_MESSAGE_QUEUE_RECEIVE
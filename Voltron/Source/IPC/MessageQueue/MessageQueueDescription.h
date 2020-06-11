//------------------------------------------------------------------------------
/// \file MessageQueueDescriptor.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref https://man7.org/linux/man-pages/man7/mq_overview.7.html
/// \brief Wrappers for message queue descriptors (mqd_t) 
//------------------------------------------------------------------------------
#ifndef IPC_MESSAGE_QUEUE_MESSAGE_QUEUE_DESCRIPTION
#define IPC_MESSAGE_QUEUE_MESSAGE_QUEUE_DESCRIPTION

#include "CreateOrOpen.h"
#include "Utilities/ErrorHandling/ErrorHandling.h"

#include <mqueue.h>
#include <string>

namespace IPC
{
namespace MessageQueue
{

class MessageQueueDescription
{
  public:

    enum class Finish : char
    {
      Manually = 'm',
      // When process is finished using the queue, close it.
      CloseOnly = 'c',
      // Close and also unlink - removing the message queue name.
      CloseAndUnlink = 'u'
    };

    //--------------------------------------------------------------------------
    /// \details Defaults to only closing the message queue descriptor, not to
    /// unlink the queue, upon destruction.
    //--------------------------------------------------------------------------
    MessageQueueDescription(
      const mqd_t message_queue_descriptor,
      const std::string& name);

    MessageQueueDescription(
      const mqd_t message_queue_descriptor,
      const std::string& name,
      const Finish finishing_method);

    mqd_t message_queue_descriptor() const
    {
      return message_queue_descriptor_;
    }

    std::string name() const
    {
      return name_;
    }

    ~MessageQueueDescription();

  private: 

    Finish finishing_method_;
    mqd_t message_queue_descriptor_;
    std::string name_;
};

class Close
{
  public:

    using OptionalErrorNumber =
      Utilities::ErrorHandling::HandleReturnValuePassively::OptionalErrorNumber;

    OptionalErrorNumber operator()(const mqd_t message_queue_descriptor);

  private:

    //--------------------------------------------------------------------------
    /// \ref https://www.man7.org/linux/man-pages/man3/mq_close.3.html
    /// \details On success, ::mq_close() returns 0; on error, -1 is returned,
    /// with errno set to indicate error.
    //--------------------------------------------------------------------------
    class HandleMqClose : public
      Utilities::ErrorHandling::HandleReturnValuePassively
    {
      public:

        using HandleReturnValuePassively::HandleReturnValuePassively;
    };
};

class Unlink
{
  public:

    using OptionalErrorNumber =
      Utilities::ErrorHandling::HandleReturnValuePassively::OptionalErrorNumber;

    OptionalErrorNumber operator()(const std::string& queue_name);  

  private:

    //--------------------------------------------------------------------------
    /// \ref https://man7.org/linux/man-pages/man3/mq_unlink.3.html
    /// \details On success, ::mq_unlink() returns 0; on error, -1 returned,
    /// with errno set to indicate error.
    /// EACCES - caller doesn't have permission to unlink this message queue.
    /// ENAMETOOLONG - argument name was too long.
    /// ENOENT - There's no message queue with given name (parameter).
    //--------------------------------------------------------------------------
    class HandleMqUnlink : public
      Utilities::ErrorHandling::HandleReturnValuePassively
    {
      public:

        using HandleReturnValuePassively::HandleReturnValuePassively;
    };
};

} // namespace MessageQueue

} // namespace IPC

#endif // IPC_MESSAGE_QUEUE_MESSAGE_QUEUE_DESCRIPTION

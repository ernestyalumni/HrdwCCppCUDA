//------------------------------------------------------------------------------
/// \file CreateOrOpen.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref http://man7.org/linux/man-pages/man2/mq_open.2.html
/// \brief Wrapper for ::mq_open, which creates a new POSIX message queue or
/// opens an existing queue.
//------------------------------------------------------------------------------
#ifndef IPC_MESSAGE_QUEUE_CREATE_OR_OPEN_H
#define IPC_MESSAGE_QUEUE_CREATE_OR_OPEN_H

#include "IPC/MessageQueue/FlagsModesAttributes.h"
#include "Utilities/ErrorHandling/ErrorHandling.h"

#include <mqueue.h> // ::mqd_t
#include <sys/stat.h> // mode_t
#include <utility>

namespace IPC
{

namespace MessageQueue
{

namespace Details
{

class HandleMqOpen : Utilities::ErrorHandling::HandleReturnValuePassively
{
  public:

    using OptionalErrorNumber =
      Utilities::ErrorHandling::HandleReturnValuePassively::OptionalErrorNumber;

    OptionalErrorNumber operator()(const mqd_t return_value);
};

} // namespace Details

class OpenConfiguration
{
  public:
    
    OpenConfiguration(const std::string& name, const int operation_flag);

    OpenConfiguration(const std::string& name, const OperationFlags flag);

    std::string name() const
    {
      return name_;
    }

    int operation_flag() const
    {
      return operation_flag_;
    }

    void add_additional_operation(const AdditionalOperationFlags flag);

  private:

    std::string name_;
    int operation_flag_;
};

//------------------------------------------------------------------------------
/// \details Wraps the following system call:
/// mqd_t mq_open(const char* name, int oflag)
/// \ref https://man7.org/linux/man-pages/man3/mq_open.3.html
//------------------------------------------------------------------------------
class NoAttributesOpen
{
  public:

    using OptionalErrorNumber =
      Utilities::ErrorHandling::HandleReturnValuePassively::OptionalErrorNumber;
    
    using OptionalMqd = std::optional<mqd_t>;

    NoAttributesOpen(const OpenConfiguration& configuration);

    std::pair<OptionalErrorNumber, OptionalMqd> operator()();

    OpenConfiguration configuration() const
    {
      return configuration_;
    }

    OptionalMqd message_queue_descriptor() const
    {
      return message_queue_descriptor_;
    }

  private:

    OpenConfiguration configuration_;
    OptionalMqd message_queue_descriptor_;
};

//------------------------------------------------------------------------------
/// \brief Creates a new POSIX message queue or opens an existing queue.
//------------------------------------------------------------------------------
class CreateOrOpen
{
  public:

    using OptionalErrorNumber =
      Utilities::ErrorHandling::HandleReturnValuePassively::OptionalErrorNumber;

    using OptionalAttributes = std::optional<Attributes>;
    using OptionalMqd = std::optional<mqd_t>;

    //--------------------------------------------------------------------------
    /// \ref http://man7.org/linux/man-pages/man2/mq_open.2.html
    /// \details mqd_t mq_open(const char* name, int oflag, mode_t mode,
    /// struct mq_attr* attr)
    /// oflag is exactly 1 of the following, O_RDONLY, O_WRONLY, O_RDWR
    /// and 0 or more additional flags.
    /// mode specifies permissions to be placed on new queue, as for ::open 
    /// (symbolic definitions for permissions bits can be obtained by including
    /// <sys/stat.h>.) Permissions settings are masked against process umask.
    //--------------------------------------------------------------------------
    CreateOrOpen(const OpenConfiguration& configuration, const mode_t mode);

    //--------------------------------------------------------------------------
    /// \details Queue is created with implementation-defined default attributes
    /// since struct mq_attr* attr argument set to NULL (via nullptr) for
    /// ::mq_open(...) call.
    //--------------------------------------------------------------------------        
    CreateOrOpen(
      const OpenConfiguration& configuration,
      const ModePermissions permission);

    //--------------------------------------------------------------------------
    /// \param maximum_message_size (in bytes).
    /// \details Fields of the struct mq_attr pointed to attr in ::mq_open(...)
    /// specify maximum number of messages and maximum size of messages that
    /// queue will allow.
    ///
    /// struct mq_attr {
    ///   long mq_flags; // Flags (ignored for mq_open())
    ///   long mq_maxmsg; // Max. # of messages on queue
    ///   long mq_msgsize; // Max. message size (bytes)
    ///   long mq_curmsgs; // # of messages currently in queue (ignored for
    ///   // mq_open())
    /// };
    /// Therefore, only the maximum number of messages, and maximum message size
    /// are needed.
    /// \ref https://man7.org/linux/man-pages/man3/mq_open.3.html
    //--------------------------------------------------------------------------   
    CreateOrOpen(
      const OpenConfiguration& configuration,
      const mode_t permission,
      const long maximum_number_of_messages,
      const long maximum_message_size);

    CreateOrOpen(
      const OpenConfiguration& configuration,
      const ModePermissions permission,
      const long maximum_number_of_messages,
      const long maximum_message_size);

    std::pair<OptionalErrorNumber, OptionalMqd> operator()();

    mode_t mode() const
    {
      return mode_;
    }

    OptionalMqd message_queue_descriptor() const
    {
      return message_queue_descriptor_;
    }

    OpenConfiguration configuration() const
    {
      return configuration_;
    }

    OptionalAttributes attributes() const
    {
      return attributes_;
    }

    void add_additional_permissions(const ModePermissions permission);

  private:

    mode_t mode_;
    OptionalMqd message_queue_descriptor_;
    OpenConfiguration configuration_;
    OptionalAttributes attributes_;
};

/*
class CreateOrOpen
{
  public:

    using OptionalErrorNumber =
      Utilities::ErrorHandling::HandleReturnValuePassively::OptionalErrorNumber;

    // Additional arguments that must be supplied if oflag is specified with
    // O_CREAT, i.e. create a new message queue since it doesn't exist.
    struct NewQueueInputs
    {
      mode_t mode_;
      long maximum_number_of_messages_;
      long maximum_message_size_;
    };

    //--------------------------------------------------------------------------
    /// \fn CreateOrOpen
    /// \ref http://man7.org/linux/man-pages/man2/mq_open.2.html
    /// \details mqd_t mq_open(const char* name, int oflag);
    /// queue is identified by name.
    /// oflag argument specifies flags that control the operation of the call.
    //--------------------------------------------------------------------------
    CreateOrOpen(const std::string& name, const int operation_flag);
*/
    //--------------------------------------------------------------------------
    /// \fn CreateOrOpen
    /// \ref http://man7.org/linux/man-pages/man2/mq_open.2.html
    /// \details mqd_t mq_open(const char* name, int oflag, mode_t mode,
    /// struct mq_attr* attr)
    /// oflag is exactly 1 of the following, O_RDONLY, O_WRONLY, O_RDWR
    /// and 0 or more additional flags.
    /// mode specifies permissions to be placed on new queue, as for ::open 
    /// (symbolic definitions for permissions bits can be obtained by including
    /// <sys/stat.h>.) Permissions settings are masked against process umask.
    ///
    /// Fields of struct mq_attr pointed to attr specify max. number of messages
    /// and max size of messages that queue will allow. This structure is
    ///
    /// struct mq_attr {
    ///   long mq_flags; /* Flags (ignored for mq_open()) */
    ///   long mq_maxmsg; /* Max. # of messages on queue */
    ///   long mq_msgsize; /* Max. message size (bytes) */
    ///   long mq_curmsgs; /* # of messages currently in queue (ignored for
    ///   mq_open(); values in remaining fileds are ignored.)
    //--------------------------------------------------------------------------
/*
    CreateOrOpen(
      const std::string& name,
      const int operation_flag,
      const mode_t mode,
      const long maximum_number_of_messages,
      const long maximum_message_size);

    CreateOrOpen(
      const std::string& name,
      const int operation_flag,
      const mode_t mode);

    std::pair<OptionalErrorNumber, std::optional<mqd_t>>
      operator()(const bool create_with_default_attributes = false);

    // Accessors

    std::string name() const
    {
      return name_;
    }

    int operation_flag() const
    {
      return operation_flag_;
    }

    // Protected to make amendable to unit tests only.
    std::optional<NewQueueInputs> new_queue_inputs() const
    {
      return new_queue_inputs_;
    }

    //--------------------------------------------------------------------------
    /// \brief Return value of ::mq_open: on success, mq_open() returns a
    /// message queue descriptor for use by other message queue functions. On
    /// error, mq_open() returns (mqd_t) -1, with errno set to indicate error.
    /// \details 
    /// EACCES Queue exists, but caller doesn't have permission to open it in
    /// specified mode.
    /// EACCES name contained more than 1 slash.
    /// EEXIST Both O_CREAT and O_EXCL were specified in oflag, but queue with
    /// this name already exists.
    /// EINVAL O_CREAT was specified in oflag, and attr wasn't NULL, but
    /// attr->mq_maxmsg or attr->mq_msqsize was invalid. Both these fields must
    /// be greater than 0. In a process that's unprivileged (doesn't have the
    /// CAP_SYS_RESOURCE capability), attr->mq_maxmsg must be less than or equal
    /// to msg_max limit, and attr->mq_msgsize must be less than or equal to the
    /// msgsize_max limit. In addition, even in a privileged process,
    /// attr->mq_maxmsg can't exceed HARD_MAX limit. (See mq_overview)
    /// EMFILE per-process limit on number of open file and message queue
    /// descriptors has been reached.
    /// ENAMETOOLONG name was too long.
    /// ENFILE system-wide limit on total number of open files and message
    /// queues has been reached.
    /// ENOENT O_CREAT flag wasn't specified in olfag, and no queue with this
    /// name exists.
    //--------------------------------------------------------------------------
    // TODO: inherit from HandleReturnValuePassively, it may fail to open
    // another queue.
    class HandleMqOpen : Utilities::ErrorHandling::HandleReturnValuePassively
    {
      public:

        OptionalErrorNumber operator()(const mqd_t return_value);
    };

  private:

    std::string name_;
    int operation_flag_;
    std::optional<NewQueueInputs> new_queue_inputs_;
};
*/

} // namespace MessageQueue
} // namespace IPC

#endif // IPC_MESSAGE_QUEUE_CREATE_OR_OPEN_H

//------------------------------------------------------------------------------
/// \file MessageQueue.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  A POSIX Message Queue as RAII with a buffer, that's a std::array.
/// \ref      
/// \details A UDP socket as RAII with a buffer, that's a std::array. 
/// \copyright If you find this code useful, feel free to donate directly and
/// easily at this direct PayPal link: 
///
/// https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
/// 
/// which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.  
/// Otherwise, I receive emails and messages on how all my (free) material on
/// physics, math, and engineering have helped students with their studies, and
/// I know what it's like to not have money as a student, but love physics (or 
/// math, sciences, etc.), so I am committed to keeping all my material 
/// open-source and free, whether or not sufficiently crowdfunded, under the 
/// open-source MIT license: feel free to copy, edit, paste, make your own
/// versions, share, use as you wish.    
/// Peace out, never give up! -EY
//------------------------------------------------------------------------------
/// COMPILATION TIPS:
///  g++ -std=c++14 ../MessageQueue_main.cpp -o -lrt ../MessageQueue_main
//------------------------------------------------------------------------------
#ifndef _MESSAGEQUEUE_H_
#define _MESSAGEQUEUE_H_

#include <cerrno> // errno
#include <cstring> // strerror
#include <fcntl.h> // For O_* constants
#include <iostream>
#include <mqueue.h>
#include <string>
#include <sys/stat.h> // For mode constants
#include <system_error>

namespace IPC
{

constexpr long maximum_number_of_messages {1000};
constexpr long maximum_message_size {100};

//------------------------------------------------------------------------------
/// \brief enum class for all flags.
/// \ref https://manpages.debian.org/testing/manpages-dev/mq_open.3.en.html
//------------------------------------------------------------------------------
enum class AllFlags : long
{
  receive_only = O_RDONLY, // Open the queue to receive messages only.
  send_only = O_WRONLY, // Open the queue to send messages only.
  send_and_receive = O_RDWR, // Open queue to both send and receive messages.
  close_on_execution = O_CLOEXEC,
  create = O_CREAT, // Create message queue if it doesn't exist
  exclusive_existence = O_EXCL, // Fail if queue with given name already exists
  nonblocking = O_NONBLOCK // non-blocking mode
};

//------------------------------------------------------------------------------
/// \brief enum class for all flags.
/// \details In <sys/stat.h>
/// \ref http://pubs.opengroup.org/onlinepubs/7908799/xsh/sysstat.h.html
//------------------------------------------------------------------------------
enum class AllModes : mode_t
{
  owner_read_write_execute = S_IRWXU, // read, write, execute/search by owner

};

std::string queue_name(const std::string& name)
{
  return "/" + name;
}

//------------------------------------------------------------------------------
/// \brief Derived class for ::mq_attr
//------------------------------------------------------------------------------
class MessageAttributes: public ::mq_attr
{
  public:

    explicit MessageAttributes(
      const long mq_flags = O_CREAT | O_EXCL | O_RDWR,
      const long mq_maxmsg = maximum_number_of_messages,
      const long mq_msgsize = maximum_message_size):
      ::mq_attr {mq_flags, mq_maxmsg, mq_msgsize}
    {}

    const unsigned int size() const 
    {
      return sizeof(::mq_attr);
    }

    ::mq_attr* to_mq_attr()
    {
      return reinterpret_cast<::mq_attr*>(this);
    }
};

//------------------------------------------------------------------------------
/// \brief POSIX Message Queue as a RAII.
/// \details 
/// \ref https://www.seas.upenn.edu/~cit595/cit595s10/lectures/ipc1.pdf
//------------------------------------------------------------------------------
class MessageQueue
{
  public:

    explicit MessageQueue(
      const std::string& queue_name):
      message_queue_descriptor_{
        open(
          queue_name.c_str(),
          static_cast<long>(AllFlags::send_and_receive) | 
            static_cast<long>(AllFlags::create) |
            static_cast<long>(AllFlags::exclusive_existence),
          static_cast<mode_t>(AllModes::owner_read_write_execute),
          message_attributes_.to_mq_attr())},
      queue_name_{queue_name}
    {}

    ~MessageQueue()
    {
      if (::mq_close(message_queue_descriptor_) < 0)
      {
//        throw std::runtime_error( // throw will always call terminate; 
        // dtors default to noexcept
          std::cout << 
            "Failed to close message queue descriptor (::mq_close) \n";
      }
    }

#if 0
    friend mqd_t open_message_queue(
      const MessageQueue message_queue,
      int oflag = static_cast<long>(AllFlags::send_only),
      mode_t mode = static_cast<mode_t>(AllModes::owner_read_write_execute))

    {
      return open(
        message_queue.queue_name(),
        oflag,
        mode,
        message_queue.to_mq_attr());
    }
#endif

    void add_to_queue(
      const char* message_ptr,
      const size_t message_length,
      const unsigned int message_priority)
    {
      message_queue_send(
        message_queue_descriptor_,
        message_ptr,
        message_length,
        message_priority);
    }

    // Accessors
    std::string queue_name() const
    {
      return queue_name_;
    }

  protected:

    mqd_t open(const std::string& name, int oflag)
    {
      const mqd_t message_queue_descriptor {
        ::mq_open(name.c_str(), oflag)
      };

      if (message_queue_descriptor < 0)
      {
        throw std::system_error(
          errno,
          std::generic_category(),
          "Failed to open message queue (::mq_open) \n");
      }
      return message_queue_descriptor;
    }

    mqd_t open(const std::string& name, int oflag, mode_t mode, mq_attr* attr)
    {
      const mqd_t message_queue_descriptor {
        ::mq_open(name.c_str(), oflag, mode, attr)
      };

      if (message_queue_descriptor < 0)
      {
        throw std::system_error(
          errno,
          std::generic_category(),
          "Failed to open message queue (::mq_open) \n");
      }
      return message_queue_descriptor;
    }

    int unlink()
    {
      const int unlink_result {::mq_unlink(queue_name_.c_str())};
      if (unlink_result < 0)
      {
        std::cout << " errno : " << std::strerror(errno) << '\n';
        throw std::system_error(
          errno,
          std::generic_category(),
          "Failed to unlink queue_name_ (::mq_link) \n");        
      }      
    }

    int close()
    {
      const int close_result {::mq_close(message_queue_descriptor_)};
      if (close_result < 0)
      {
        std::cout << " errno : " << std::strerror(errno) << '\n';
        throw std::system_error(
          errno,
          std::generic_category(),
          "Failed to close message queue descriptor (::mq_close) \n");        
      }
      return close_result;
    }

    // Accessors
    ::mq_attr* to_mq_attr()
    {
      return message_attributes_.to_mq_attr();
    }

    const mqd_t message_queue_descriptor() const
    {
      return message_queue_descriptor_;
    }

  private:

    void message_queue_send(
      const mqd_t mqdes,
      const char* msg_ptr,
      size_t msg_len,
      unsigned int msg_prio)
    {
      if (::mq_send(mqdes, msg_ptr, msg_len, msg_prio) < 0)
      {
        std::cout << " errno : " << std::strerror(errno) << '\n';
        throw std::system_error(
          errno,
          std::generic_category(),
          "Failed to send message (::mq_send) to message queue descriptor.\n");        
      }
    }

    void message_queue_receive(
      const mqd_t mqdes, 
      char *msg_ptr,
      const size_t msg_len,
      unsigned int* msg_prio)
    {
      if (::mq_receive(mqdes, msg_ptr, msg_len, msg_prio) < 0)
      {
        std::cout << " errno : " << std::strerror(errno) << '\n';
        throw std::system_error(
          errno,
          std::generic_category(),
          "Failed to receive message (::mq_receive) to message queue descriptor.\n"); 
      }
    }

    mqd_t message_queue_descriptor_;
    std::string queue_name_;
    MessageAttributes message_attributes_;

};



} // namespace IPC

#endif // _MESSAGEQUEUE_H_

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
///  g++ -std=c++14 FileOpen_main.cpp FileOpen.cpp -o FileOpen_main
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
        ::mq_open(
          queue_name.c_str(),
          static_cast<mode_t>(AllModes::owner_read_write_execute)
        )},
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

  protected:

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

  private:

    mqd_t message_queue_descriptor_;
    std::string queue_name_;
};



} // namespace IPC

#endif // _MESSAGEQUEUE_H_

//------------------------------------------------------------------------------
/// \file EpollFd.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief Create and encapsulate an epoll fd.
/// \ref http://man7.org/linux/man-pages/man2/epoll_create.2.html
/// http://cs241.cs.illinois.edu/coursebook/Networking#non-blocking-io
//------------------------------------------------------------------------------
#ifndef IO_EPOLL_EPOLL_FD_H
#define IO_EPOLL_EPOLL_FD_H

#include <sys/epoll.h>

#include "Utilities/ErrorHandling/ErrorHandling.h"

namespace IO
{
namespace Epoll
{

//------------------------------------------------------------------------------
/// \brief enum class for all flags for the creation of a new epoll instance.
//------------------------------------------------------------------------------
enum class EpollFlags : int
{
  default_value = 0,
  close_on_execute = EPOLL_CLOEXEC
};

//------------------------------------------------------------------------------
/// \brief Create a new epoll instance.
//------------------------------------------------------------------------------
class EpollFd
{
  public:

  	explicit EpollFd(int size, const bool close_upon_destruction = true);

  	explicit EpollFd(
      const EpollFlags flags = EpollFlags::default_value,
      const bool close_upon_destruction = true);

  	virtual ~EpollFd();

  	int fd() const
  	{
  	  return fd_;
  	}

  protected:

    //--------------------------------------------------------------------------
    /// \ref http://man7.org/linux/man-pages/man2/epoll_create.2.html
    /// \brief Return value of ::epoll_create, ::epoll_create1, on success, is a
    /// file descriptor (a nonnegative integer).
    /// On error, -1 returned, and errno set.
    /// EINVAL size is not positive.
    /// EINVAL (epoll_create1()) Invalid value specified in flags.
    /// EMFILE per-user limit on number of epoll instances imposed by
    /// /proc/sys/fs/epoll/max_user_instances encountered.
    /// EMFILE per-process limit on number of open fds reached.
    /// ENFILE system-wide limit on the total number of open files has been
    /// reached.
    /// ENOMEM insufficient memory to create kernel object.
    //--------------------------------------------------------------------------
		class HandleEpollCreate :
      public Utilities::ErrorHandling::HandleReturnValuePassively
		{
			public:

				HandleEpollCreate();

				//void operator()(const int result_value);

			private:

				//using HandleReturnValuePassively::operator();
		};

    int create_epoll_with_size_hint(int size);

    int create_epoll(const EpollFlags flags);

  private:

  	int fd_;
    bool close_upon_destruction_;
};

} // namespace Epoll
} // namespace IO

#endif // IO_EPOLL_EPOLL_FD_H

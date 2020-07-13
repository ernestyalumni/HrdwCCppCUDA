//------------------------------------------------------------------------------
/// \file EpollFd.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief Create a new epoll instance and encapsulate it.
/// \ref http://man7.org/linux/man-pages/man2/epoll_create.2.html
/// http://cs241.cs.illinois.edu/coursebook/Networking#non-blocking-io
//------------------------------------------------------------------------------
#include "EpollFd.h"

#include "Cpp/Utilities/TypeSupport/UnderlyingTypes.h"
#include "Utilities/ErrorHandling/ErrorHandling.h"

#include <sys/epoll.h>
#include <unistd.h> // ::close

using Cpp::Utilities::TypeSupport::get_underlying_value;
using Utilities::ErrorHandling::HandleClose;

namespace IO
{
namespace Epoll
{

EpollFd::EpollFd(int size, const bool close_upon_destruction):
	fd_{create_epoll_with_size_hint(size)},
  close_upon_destruction_{close_upon_destruction}
{}

EpollFd::EpollFd(const EpollFlags flags, const bool close_upon_destruction):
	fd_{create_epoll(flags)},
  close_upon_destruction_{close_upon_destruction}
{}

EpollFd::~EpollFd()
{
  if (close_upon_destruction_)
  {
  	int return_value {::close(fd_)};
  	HandleClose()(return_value);
  }
}

EpollFd::HandleEpollCreate::HandleEpollCreate() = default;

void EpollFd::HandleEpollCreate::operator()(const int result_value)
{ 
  this->operator()(
  	result_value,
  	"create new epoll instance (::epoll_create or ::epoll_create1)");
}

int EpollFd::create_epoll_with_size_hint(int size)
{
	int return_value {::epoll_create(size)};

	HandleEpollCreate()(return_value);

	return return_value;
}

int EpollFd::create_epoll(const EpollFlags flags)
{
	int return_value {::epoll_create1(get_underlying_value(flags))};

	HandleEpollCreate()(return_value);

	return return_value;
}


} // namespace Epoll
} // namespace IO

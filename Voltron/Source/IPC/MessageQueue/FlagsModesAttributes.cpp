//------------------------------------------------------------------------------
/// \file DataStructures.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref http://man7.org/linux/man-pages/man2/mq_open.2.html
/// \brief Wrapper for message queue flags and attributes.
//------------------------------------------------------------------------------
#include "DataStructures.h"

#include <ostream>

namespace IPC
{
namespace MessageQueue
{

std::ostream& Attributes::operator<<(
  std::ostream& os,
  const Attributes& attributes)
{
  os << attributes.mq_flags << ' ' << attributes.mq_maxmsg << ' ' <<
    attributes.mq_msgsize << ' ' << attributes.mq_curmsgs << '\n';

  return os;
}

} // namespace MessageQueue
} // namespace IPC

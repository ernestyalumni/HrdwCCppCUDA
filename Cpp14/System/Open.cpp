//------------------------------------------------------------------------------
/// \file Open.cpp
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief ::open as a C++ functor source file.
/// \ref http://man7.org/linux/man-pages/man2/open.2.html
/// \details Open and possibly create a file.
/// \copyright If you find this code useful, feel free to donate directly
/// (username ernestyalumni or email address above), going directly to:
///
/// paypal.me/ernestyalumni
///
/// which won't go through a 3rd. party like indiegogo, kickstarter, patreon.
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
/// g++ -std=c++17 -I ../ Open.cpp ../Utilities/Errno.cpp
///   ../Utilities/ErrorHandling.cpp Open_main.cpp -o Open_main
//------------------------------------------------------------------------------
#include "Open.h"

#include "Utilities/casts.h" // get_underlying_value

#include <string>

using Utilities::get_underlying_value;

namespace System
{

Open::Open(const std::string& pathname, const AccessModes& access_mode):
  pathname_{pathname},
  access_mode_{access_mode},
  flags_{get_underlying_value<AccessModes>(access_mode_)},
  mode_{},
  more_creation_flags_{},
  more_status_flags_{},
  more_modes_{}
{}

Open::Open(const std::string& pathname, const int flags):
  pathname_{pathname},
  access_mode_{},
  flags_{flags},
  mode_{},
  more_creation_flags_{},
  more_status_flags_{},
  more_modes_{}
{}

Open::Open(
  const std::string& pathname,
  const AccessModes& access_mode,
  const mode_t mode
  ):
  pathname_{pathname},
  access_mode_{access_mode},
  flags_{},
  mode_{mode},
  more_creation_flags_{},
  more_status_flags_{},
  more_modes_{}
{}

Open::Open(
  const std::string& pathname,
  const int flags,
  const mode_t mode
  ):
  pathname_{pathname},
  access_mode_{},
  flags_{flags},
  mode_{mode},
  more_creation_flags_{},
  more_status_flags_{},
  more_modes_{}
{}

int Open::operator()(const bool with_mode)
{
  int result;

  if (with_mode)
  {
    result = ::open(pathname_.c_str(), flags_, mode_);
  }
  else
  {
    result = ::open(pathname_.c_str(), flags_);
  }

  HandleOpen()(result);

  return result;
}

void Open::add_creation_flag(const CreationFlags& creation_flag)
{
  more_creation_flags_.emplace_back(creation_flag);

  flags_ = flags_ | get_underlying_value<CreationFlags>(creation_flag);
}

void Open::add_status_flag(const StatusFlags& status_flag)
{
  more_status_flags_.emplace_back(status_flag);

  flags_ = flags_ | get_underlying_value<StatusFlags>(status_flag);
}

void Open::add_mode(const Modes& mode)
{
  more_modes_.emplace_back(mode);

  mode_ = mode_ | get_underlying_value<Modes>(mode);
}

Open::HandleOpen::HandleOpen() = default;

void Open::HandleOpen::operator()(const int result)
{
  this->operator()(result, "Open pathname or file (::open)");
}

} // namespace System
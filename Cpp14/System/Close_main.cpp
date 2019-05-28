//------------------------------------------------------------------------------
/// \file Close_main.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief Main driver file for ::close as a C++ functor 
/// \ref http://man7.org/linux/man-pages/man2/close.2.html
/// \details Close and possibly create a file.
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
///  g++ -std=c++17 -I ../ Close.cpp Fork.cpp Open.cpp ../Utilities/Errno.cpp ../Utilities/ErrorHandling.cpp Close_main.cpp -o Close_main
//------------------------------------------------------------------------------
#include "Close.h"
#include "Fork.h"
#include "Open.h"

using System::AccessModes;
using System::CreationFlags;
using System::Fork;
using System::Modes;
using System::Open;
using System::Close;

#include <iostream>

int main()
{

  {
    Open open {"abc.txt", AccessModes::write_only};
    open.add_mode(Modes::user_read);
    open.add_mode(Modes::user_write);
    open.add_mode(Modes::group_read);
    open.add_mode(Modes::group_write);
    open.add_mode(Modes::others_read);
    open.add_mode(Modes::others_write);

    open.add_creation_flag(CreationFlags::create);
    open.add_creation_flag(CreationFlags::truncate);      

    int fd {open()};

    std::cout << " fd : " << fd;

    Fork fork;

    fork();

    // On success, PID of child process returned in the parent, and 0 returned
    // in the child.
    std::cout << " fork.process_id() : " << fork.process_id() << '\n';

    std::cout << " fd : " << fd;

    Close()(fd);

//    Close()(fd);
  }

}
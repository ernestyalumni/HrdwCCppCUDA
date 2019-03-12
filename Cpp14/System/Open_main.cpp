//------------------------------------------------------------------------------
/// \file Open_main.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief Main driver file for ::open as a C++ functor 
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
///  g++ -std=c++17 -I ../ Open.cpp ../Utilities/Errno.cpp ../Utilities/ErrorHandling.cpp Open_main.cpp -o Open_main
//------------------------------------------------------------------------------
#include "Open.h"

#include <iostream>
#include <sys/fcntl.h>
#include <unistd.h>

using System::Open;
using System::AccessModes;
using System::CreationFlags;
using System::StatusFlags;
using System::Modes;

int main()
{
  // mode_tProperties
  {
    std::cout << "\n mode_tProperties\n";
    std::cout << " sizeof(mode_t) : " << sizeof(mode_t) << '\n'; // 4 bytes
  }

  // OpenConstructs
  {
    std::cout << "\n OpenConstrusts\n";

    Open open {"abc.txt", O_WRONLY | O_CREAT | O_TRUNC, 0666};

    std::cout << open.flags() << '\n';
    std::cout << (O_WRONLY | O_CREAT | O_TRUNC) << '\n';
    std::cout << open.mode() << '\n';
    std::cout << (open.mode() == 0666) << '\n';   

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
      //open.add_creation_flag(CreationFlags::temporary_file);      

      std::cout << " open.flags() : " << open.flags() << '\n';
      std::cout << (O_WRONLY | O_CREAT | O_TRUNC) << '\n';
      std::cout << " open.mode() : " << open.mode() << '\n';
      std::cout << (open.mode() == 0666) << '\n';   

      int fd {open()};

      std::cout << " fd : " << fd << '\n';

      ::close(fd);
    }

    {
      int fd {::open("abc.txt", O_WRONLY | O_CREAT | O_TRUNC, 0666)};

      std::cout << " fd : " << fd << '\n';

      ::close(fd);
    }

    {
      std::string test_string {"abc.txt"};

      int fd {::open(test_string.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0666)};

      std::cout << " fd : " << fd << '\n';

      ::close(fd);
    }


    {
      Open open {"abc.txt", O_WRONLY | O_CREAT | O_TRUNC, 0666};

      std::cout << " open.pathname() : " << open.pathname() << '\n';
      std::cout << " open.pathname().c_str() : " << open.pathname().c_str() <<
        '\n';

      int fd {::open(open.pathname().c_str(), open.flags(), open.mode())};

      std::cout << " fd : " << fd << '\n';

      ::close(fd);
    }

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

      ::fork();

      ::write(fd, "xyz", 3);

      printf("This is what lseek does : %ld\n", ::lseek(fd, 0, SEEK_CUR));

      ::close(fd);


    }

  }

}
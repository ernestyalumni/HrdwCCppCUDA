//------------------------------------------------------------------------------
/// \file EventFd_main.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Main driver file for eventfd, for event notification.
/// \ref http://man7.org/linux/man-pages/man2/eventfd.2.html  
/// \details 
/// \copyright If you find this code useful, feel free to donate directly via
/// PayPal (username ernestyalumni or email address above); my PayPal profile:
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
///  g++ -std=c++14 EventFd_main.cpp -o EventFd_main
//------------------------------------------------------------------------------
#include "EventFd.h"

#include "../Utilities/Chrono.h"

#include <cstdlib> // std::strtoull 
#include <iostream>
#include <thread>
#include <unistd.h> // fork

using IO::EventFdFlags;
using IO::EventFd;

using namespace Utilities::Literals;

// \url https://docs.microsoft.com/en-us/cpp/cpp/parsing-cpp-command-line-arguments?view=vs-2017
int main(
  int argc, // Number of strings in array argv
  char* argv[]) // Array of command-line argument strings
{
  // EventFdFlagsIsAnEnumClass
  {
    std::cout << "\n EventFdFlagsIsAnEnumClass \n";  
    std::cout << " EventFdFlags::default_value : " << 
      static_cast<int>(EventFdFlags::default_value) << '\n'; // 0
    std::cout << " EventFdFlags::close_on_execute : " << 
      static_cast<int>(EventFdFlags::close_on_execute) << '\n'; // 524288
    std::cout << " EventFdFlags::non_blocking : " << 
      static_cast<int>(EventFdFlags::non_blocking) << '\n'; // 2048
    std::cout << " EventFdFlags::semaphore : " << 
      static_cast<int>(EventFdFlags::semaphore) << '\n'; // 1
  }

  // Sample Program
  // \ref https://linux.die.net/man/2/eventfd
  {
    std::cout << "\n Sample program : " << '\n';

    EventFd<> event_fd {0};

    event_fd.set_buffer(1);
    event_fd.write();
    event_fd.set_buffer(2);
    event_fd.write();
    event_fd.set_buffer(4);
    event_fd.write();
    event_fd.set_buffer(7);
    event_fd.write();

    std::this_thread::sleep_for(2s);

    event_fd.read();
    std::cout << " eventfd.buffer() : " << event_fd.buffer() << '\n';

  }

  // Sample Program
  // \url http://man7.org/linux/man-pages/man2/eventfd.2.html
  {
    if (argc < 2)
    {
      std::cerr << "Usage: " << argv[0] << " <num>...\n";
      exit(EXIT_FAILURE);
    }

    EventFd<> event_fd {0};

    // \url http://man7.org/linux/man-pages/man2/fork.2.html
    // \details fork() creates a new process by duplicating the calling
    // process. The new process is referred to as child process.
    // On success, pid of child process is returned in parent,
    // 0 is returned in the child.
    switch (fork())
    {
      case 0:

        for (int j {1}; j < argc; j++)
        {
          std::cout << "Child writing " << argv[j] << " to efd \n";

          // Interprets unsigned integer value in a byte string pointed to by
          // str
          uint64_t u = std::strtoull(argv[j], nullptr, 0); // std::strtoull()
          // allows various bases
          event_fd.set_buffer(u);
          event_fd.write();
        }

        std::cout << "Child completed write loop\n";

        exit(EXIT_SUCCESS);

      default:

        std::this_thread::sleep_for(2s);

        std::cout << "Parent about to read\n";

        event_fd.read();

        std::cout << " Parent read " << event_fd.buffer() << '\n';

        exit(EXIT_SUCCESS);
    }
  }

}

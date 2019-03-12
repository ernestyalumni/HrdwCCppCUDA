//------------------------------------------------------------------------------
/// \file TemporaryFiles.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief Temporary file system calls as a C++ functor 
/// \ref https://www.thegeekstuff.com/2012/06/c-temporary-files 
/// http://man7.org/linux/man-pages/man3/mkstemp.3.html
/// \details Generate temporary file names.
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
///  g++ -std=c++17 -I ../ Open.cpp ../Utilities/Errno.cpp
///   ../Utilities/ErrorHandling.cpp Open_main.cpp -o Open_main
//------------------------------------------------------------------------------
#ifndef _SYSTEM_TEMPORARY_FILES_H_
#define _SYSTEM_TEMPORARY_FILES_H_

#include <cstdlib> // <stdlib.h>
#include <string>

#include "Utilities/ErrorHandling.h" // HandleReturnValue

namespace System
{

//------------------------------------------------------------------------------
/// \class MakeTemporary
/// \brief Generates a unique temporary filename from template, creates and
/// opens the file, and returns an open fd for the file.
/// \details
/// #include <stdlib.h>
/// int mkstemp(char* template)
/// \ref https://www.thegeekstuff.com/2012/06/c-temporary-files
/// \url http://man7.org/linux/man-pages/man3/mkstemp.3.html
//------------------------------------------------------------------------------
class MakeTemporary
{
  public:

    MakeTemporary();

    MakeTemporary(const std::string& template);
};

} // namespace System

#endif // _SYSTEM_TEMPORARY_FILES_H_


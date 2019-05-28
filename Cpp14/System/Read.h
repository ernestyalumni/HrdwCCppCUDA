//------------------------------------------------------------------------------
/// \file Read.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief read() as a C++ functor with CRTP pattern.
/// \ref http://man7.org/linux/man-pages/man2/read.2.html
/// \details Read from a fd.
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
///  g++ -std=c++17 -I ../ Close.cpp ../Utilities/Errno.cpp
///   ../Utilities/ErrorHandling.cpp Close_main.cpp -o Close_main
//------------------------------------------------------------------------------
#ifndef _SYSTEM_READ_H_
#define _SYSTEM_READ_H_

#include "Utilities/ErrorHandling.h" // HandleReturnValue

namespace System
{

//------------------------------------------------------------------------------
/// \class Read
/// \brief ::read() system call wrapped in a C++ functor, with CRTP pattern
/// \details
///
/// #include <unistd.h>
/// ssize_t ::read(int fd, void* buf, size_t count);
///
/// ::read() attempts to read up to count bytes from fd into the buffer starting
/// at buf.
///
/// On files that support seeking, read operation commences at file offset, and
/// the file offset is incremented by number of bytes read. If file offset is at
/// or past the end of file, no bytes are read, and read() returns 0.
//------------------------------------------------------------------------------
class Read
{
  public:

    Read();

    ssize_t operator()(const int fd, void* buf, const size_t count);

  protected:

    //--------------------------------------------------------------------------
    /// \class HandleWrite
    /// \brief On success, the number of bytes written is returned. On error, -1
    /// is returned, and errno set to indicate cause of error.
    ///
    /// Note that successful write() may transfer fewer than count bytes.
    //--------------------------------------------------------------------------
    class HandleRead : public Utilities::ErrorHandling::HandleReturnValue
    {
      public:

        HandleRead();

        void operator()(const ssize_t result);

    };

  private:

    ssize_t last_number_of_bytes_to_read_;
}; // class Read

//template <typename Implementation>
//class Read
//{
//  public:

//    Read() = default;

//    virtual ~Read() = default;

//    ssize_t operator()(const int fd, const std::size_t count)
//    {
//      return static_cast<Implementation&>(*this)(fd, count);
//    }
//}; // class Read

//template 
//class ReadBuffer : public Read<ReadBuffer>
//{

//};

} // namespace System

#endif // _SYSTEM_READ_H_
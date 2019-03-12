//------------------------------------------------------------------------------
/// \file Open.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief ::open as a C++ functor 
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
///  g++ -std=c++17 -I ../ Open.cpp ../Utilities/Errno.cpp
///   ../Utilities/ErrorHandling.cpp Open_main.cpp -o Open_main
//------------------------------------------------------------------------------
#ifndef _SYSTEM_OPEN_H_
#define _SYSTEM_OPEN_H_

#include <sys/fcntl.h> // O_RDONLY, O_WRONLY, O_RDWR, O_CLOEXEC, O_CREAT,... 
#include <vector>

#include "Utilities/ErrorHandling.h" // HandleReturnValue

namespace System
{

//------------------------------------------------------------------------------
/// \brief Enum class for access modes; argument flags for ::open must include
/// one of the following access modes.
//------------------------------------------------------------------------------
enum class AccessModes: int
{
  read_only = O_RDONLY,
  write_only = O_WRONLY,
  read_and_write = O_RDWR
};

//------------------------------------------------------------------------------
/// \brief Enum class for creation flags; can be bitwise-or'd in flags.
/// \url http://man7.org/linux/man-pages/man2/open.2.html
/// \details Use of this flag (O_CLOEXEC) is essential in some multithreaded
/// programs, because using a separate ::fcntl F_SETFD operation to set
/// FD_CLOEXEC flag doesn't suffice to avoid race conditions where 1 thread
/// opens a fd and attempts to set its close-on-exec flag using ::fcntl, at the
/// same time as a another thread does ::fork plus ::execve.
/// O_NOCTTY - If pathname refers to a terminal device-see tty(4)-t won't become
/// process's controlling terminal even if process doesn't have one.
/// O_NOFOLLOW - If pathname is a symbolic link, then open fails, with error
/// ELOOP. Symbolic links in earlier components of pathname will still be
/// followed.
/// O_TMPFILE - Create unnamed temporary regular file. Pathname argument
/// specifies a directory; an unnamed inode will be created in that directory's
/// filesystem. Anything written to resulting file will be lost when last fd is
/// closed, unless file is given a name.
/// O_TRUNC - If file already exists and is a regular file and access mode
/// allows writing (i.e., is O_RDWR or OWRONLY) it'll truncate to length 0.
//------------------------------------------------------------------------------
enum class CreationFlags : int
{
  close_on_execution = O_CLOEXEC,
  create = O_CREAT, // If pathname doesn't exist, create it as a regular file.
  directory = O_DIRECTORY, // If pathname isn't a directory, cause open to fail.
  no_controlling_terminal = O_NOCTTY,
  no_symbolic_link = O_NOFOLLOW,
  temporary_file = O_TMPFILE, // Create an unnamed temporary regular file.
  truncate = O_TRUNC
};

//------------------------------------------------------------------------------
/// \brief Enum class for file status flags.
/// O_ASYNC - enable signal-driven I/O: generate a signal (SIGIO by default, but
/// this can be changed via ::fcntl) when input or output becomes possible on
/// this fd.
/// O_PATH - Obtain fd used for 2 purposes: to indicate location in filesystem
/// tree and to perform operations that act purely at fd level.
/// O_SYNC - Write operations on file will complete according to requirements of
/// synchronized I/O file integrity completion (by constrast with O_DSYNC,
/// synchronized I/O data integrity completion).
//------------------------------------------------------------------------------
enum class StatusFlags : int
{
  append = O_APPEND, // file is opened in append mode
  asynchronous = O_ASYNC,
  direct = O_DIRECT, // try to minimize cache effects of I/O to and from file.
  data_synchronize = O_DSYNC,
  exclusive = O_EXCL, // ensure call creates file.
  large_file = O_LARGEFILE,
  no_access_time = O_NOATIME, // Don't update file last access time.
  nonblocking = O_NONBLOCK,
  path = O_PATH,
  synchronize = O_SYNC 
};

//------------------------------------------------------------------------------
/// \brief Enum class for mode argument
/// \details mode argument specifies file mode bits be applied when a new file
/// is created. Argument must be supplied when O_CREAT or O_TMPFILE is specified
/// in flags; if neither O_CREAT nor O_TMPFILE is specified, then mode's
/// ignored.
/// Effective mode is modified by process's umask in usual way: in absence of a
/// default ACL, mode of created file is (mode & ~umask).
//------------------------------------------------------------------------------
enum class Modes : mode_t
{
  user_read_write_execute = S_IRWXU, // 00700 used (file owner) has rwx
  user_read = S_IRUSR, // 00400
  user_write = S_IWUSR, // 00200
  user_execute = S_IXUSR, // 00100
  group_read_write_execute = S_IRWXG, // 00070
  group_read = S_IRGRP, // 00040
  group_write = S_IWGRP, // 00020
  group_execute = S_IXGRP, // 00010
  others_read_write_execute = S_IRUSR, // 00007
  others_read = S_IROTH, // 00004
  others_write = S_IWOTH, // 00002 
  others_execute = S_IXOTH  // 00001
};

//------------------------------------------------------------------------------
/// \class Open
/// \brief ::open() system call wrapped in a C++ functor.
/// \details 
/// 
/// int open(const char* pathname, int flags);
/// int open(const char* pathname, int flags, mode_t mode);
///
/// ::open system call opens file specified by pathname. If specified
/// file doesn't exist, it may optionally (if O_CREATE is specified in flags) be
/// created by ::open().
/// Return value of ::open() is a fd, small, nonnegative integer that's used in
/// subsequent system calls (::read, ::write, ::lseek, ::fcntl, etc.) to refer
/// to the open file.
/// - fd returned by successful call will be lowest-numbered fd not currently
///   open for process.
/// By default, new fd is set to remain open across an ::execve (i.e. FD_CLOEXEC
/// fd flag described in ::fcntl is initially disabled).
/// - O_CLOEXEC flag can be used to change this default. File offset is set to
///   the beginning of the file (see ::lseek)
/// Call to ::open() creates a new *open file description*, an entry in the
/// system-wide table of open files.
/// - open file description records the file offset and file status flags.
/// - fd is a reference to an open file description; this reference is
///   unaffected if *pathname* is subsequently removed or modified to refer to a
///   different file.
//------------------------------------------------------------------------------
class Open
{
  public:

    Open() = delete;

    Open(const std::string& pathname, const AccessModes& access_mode);

    Open(const std::string& pathname, const int flags);

    Open(
      const std::string& pathname,
      const AccessModes& access_mode,
      const mode_t mode);

    Open(
      const std::string& pathname,
      const int flags,
      const mode_t mode);

    int operator()(const bool with_mode = true);

    // Accessors

    std::string pathname() const
    {
      return pathname_;
    }

    AccessModes access_mode() const
    {
      return access_mode_;
    }

    int flags() const
    {
      return flags_;
    }

    mode_t mode() const
    {
      return mode_;
    }

    std::vector<CreationFlags> more_creation_flags() const
    {
      return more_creation_flags_;
    }

    std::vector<StatusFlags> more_status_flags() const
    {
      return more_status_flags_;
    }

    std::vector<Modes> more_modes() const
    {
      return more_modes_;
    }

    // Setters

    void add_creation_flag(const CreationFlags& creation_flag);
    void add_status_flag(const StatusFlags& status_flag);
    void add_mode(const Modes& mode);

  protected:

    //--------------------------------------------------------------------------
    /// \class HandleOpen
    /// 
    //--------------------------------------------------------------------------
    class HandleOpen : public Utilities::ErrorHandling::HandleReturnValue
    {
      public:

        HandleOpen();

        void operator()(const int result);

      private:

        using HandleReturnValue::operator();
    };

  private:

    std::string pathname_;
    AccessModes access_mode_;
    int flags_;
    mode_t mode_;
    std::vector<CreationFlags> more_creation_flags_;
    std::vector<StatusFlags> more_status_flags_;
    std::vector<Modes> more_modes_;
};

} // namespace System

#endif // _SYSTEM_OPEN_H_
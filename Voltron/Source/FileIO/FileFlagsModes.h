//------------------------------------------------------------------------------
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \ref https://man7.org/linux/man-pages/man2/open.2.html
/// \brief Flags for ::open
//------------------------------------------------------------------------------
#ifndef FILE_IO_FILE_FLAGS_MODES_H
#define FILE_IO_FILE_FLAGS_MODES_H

#include <fcntl.h> // O_RDONLY, ...
#include <sys/types.h> // O_*

namespace FileIO
{

//------------------------------------------------------------------------------
/// \details argument flags for ::open() must include 1 of following access
/// modes.
/// \ref https://man7.org/linux/man-pages/man2/open.2.html
//------------------------------------------------------------------------------
enum class AccessMode : int
{
  read_only = O_RDONLY,
  write_only = O_WRONLY,
  read_write = O_RDWR
};

//------------------------------------------------------------------------------
/// \brief Zero or more file creation flags and file status flags can be bitwise
/// or'd for flags in ::open.
/// \ref https://man7.org/linux/man-pages/man2/open.2.html
//------------------------------------------------------------------------------
enum class FileFlag : int
{
  // When possible, file is opened in nonblocking mode. Neither ::open() nor any
  // subsequent I/O operations on fd which is returned will cause the calling
  // process to wait.
  nonblocking_open = O_NONBLOCK,
  // Write operations on file will complete according to requirements of
  // synchronized I/O file integrity completion (by contrast with synchronized
  // I/O data integrity completion provided by O_DSYNC).
  synchronize_writes = O_SYNC,
};

int to_access_mode_value(const AccessMode access_mode);

int to_file_flag_value(const AccessMode access_mode);

} // namespace FileIO

#endif // FILE_IO_FILE_FLAGS_MODES_H

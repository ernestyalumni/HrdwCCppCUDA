#ifndef FILE_IO_GET_FILE_SIZE_H
#define FILE_IO_GET_FILE_SIZE_H

#include <ios> // std::streamsize
#include <string>
#include <sys/stat.h>

namespace FileIO
{

//------------------------------------------------------------------------------
/// \details std::streamsize is an implementation-defined signed integral type
/// to represent number of characters transferred in an I/O operation or size of
/// I/O buffer.
//------------------------------------------------------------------------------
std::streamsize get_file_size(const std::string& file_path);

class GetFileStatus
{
  public:

    GetFileStatus(const std::string& file_path);

    ::off_t get_file_size();

    struct stat buffer_;
};

} // namespace FileIO

#endif // FILE_IO_GET_FILE_SIZE_H

#include "GetFileSize.h"

#include <fstream> // std::ifstream
#include <ios> // std::ios, std::streamsize
#include <limits> // std::numeric_limits
#include <stdexcept>
#include <string>

namespace FileIO
{

std::streamsize get_file_size(const std::string& file_path)
{
  std::ifstream file {};
  file.open(file_path, std::ios::in | std::ios::binary);

  if (file.is_open())
  {
    //--------------------------------------------------------------------------
    /// member function, max - returns largest finite value of given type.
    /// ignore - extract and discards characters from input stream until and
    /// including delim. Extracts characters from stream and discards them until
    /// any of the following conditions occurs: count characters extracted. But
    /// since count equals std::numeric_limits<std::streamsize>::max(), then
    /// this test is disabled.
    /// So EOF (end of file) conditions.
    //--------------------------------------------------------------------------
    file.ignore(std::numeric_limits<std::streamsize>::max());

    // gcount - returns number of characters extracted by last unformatted input
    // operation.
    std::streamsize length {file.gcount()};
    // clear - sets stream error state flags by assigning them the value of
    // state; default value being goodbit().
    file.clear();

    // Sets input position indicator. beg - beginning of stream, 0 is relative
    // position to set input position indicator to.
    file.seekg(0, std::ios_base::beg);

    return length;
  }
  else
  {
    return static_cast<std::streamsize>(-1);
  }
}

GetFileStatus::GetFileStatus(const std::string& file_path):
  buffer_{}
{
  if (::stat(file_path.c_str(), &buffer_) != 0)
  {
    throw std::runtime_error("Failed to run stat on " + file_path);
  }
}

::off_t GetFileStatus::get_file_size()
{
  return buffer_.st_size;
}

} // namespace FileIO

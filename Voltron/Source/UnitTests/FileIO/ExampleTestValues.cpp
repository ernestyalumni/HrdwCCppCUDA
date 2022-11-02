#include "ExampleTestValues.h"

#include <filesystem>
#include <string>

namespace UnitTests
{
namespace FileIO
{

std::string example_file_path_1()
{
  // Returns absolute path of current working directory, obtained as if (in
  // native format) by POSIX.
  std::string current_path_1 {std::filesystem::current_path()};

  return current_path_1 + "/../Source/UnitTests/FileIO/Data/P1test.txt";
}

} // namespace FileIO
} // namespace UnitTests
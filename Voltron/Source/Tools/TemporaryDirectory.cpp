#include "TemporaryDirectory.h"

#include <boost/filesystem.hpp>
#include <unistd.h> // ::get_current_dir_name
#include <string>

using std::string;

namespace Tools
{

TemporaryDirectory::TemporaryDirectory(
  const string& directory_name_prefix,
  const string& base_directory_path)
{
  path_ = make_temporary_directory(directory_name_prefix, base_directory_path);
}

TemporaryDirectory::TemporaryDirectory(const string& directory_name_prefix)
{
  // https://linux.die.net/man/3/get_current_dir_name
  // char* get_current_dir_name(void);
  // Returns null-terminated string containing absolute pathname that's current
  // working directory of the calling process.
  char* current_dir_name_arr {::get_current_dir_name()};
  string current_dir_name_str {current_dir_name_arr};

  path_ = make_temporary_directory(directory_name_prefix, current_dir_name_str);

  free(current_dir_name_arr);
}

TemporaryDirectory::~TemporaryDirectory()
{
  // cf. https://www.boost.org/doc/libs/1_45_0/libs/filesystem/v3/doc/reference.html
  boost::filesystem::remove_all(path_);
}

std::string TemporaryDirectory::make_temporary_directory(
  const string& directory_name_prefix,
  const string& base_directory_path)
{
  string template_string {
    base_directory_path + "/" + directory_name_prefix + "XXXXXX"};

  // cf. http://man7.org/linux/man-pages/man3/mkdtemp.3.html
  // RETURN VALUE
  // mkdtemp() function returns a pointer to the modified template string on
  // success, and NULL on failure, in which case errno is set appropriately.
  string modified_template {::mkdtemp(template_string.data())};

  return modified_template;
}

string create_temporary_filename(
  const TemporaryDirectory& temp_dir,
  const string& filename)
{
  return temp_dir.path() + "/" + filename;
}


} // namespace Tools
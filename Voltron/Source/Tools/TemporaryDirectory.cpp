#include "TemporaryDirectory.h"

#include <boost/filesystem.hpp>
#include <string>

namespace Tools
{

TemporaryDirectory::TemporaryDirectory(
  const std::string& directory_name_prefix,
  const std::string& base_directory_path)
{
  path_ = make_temporary_directory(directory_name_prefix, base_directory_path);
}

TemporaryDirectory::TemporaryDirectory(const std::string& directory_name_prefix)
{
  std::string current_dir_name_str {::get_current_dir_name()};

  path_ = make_temporary_directory(directory_name_prefix, current_dir_name_str);
}

TemporaryDirectory::~TemporaryDirectory()
{
  // cf. https://www.boost.org/doc/libs/1_45_0/libs/filesystem/v3/doc/reference.html
  boost::filesystem::remove_all(path_);
}

std::string TemporaryDirectory::make_temporary_directory(
  const std::string& directory_name_prefix,
  const std::string& base_directory_path)
{
  std::string template_string {
    base_directory_path + "/" + directory_name_prefix + "XXXXXX"};

  // cf. http://man7.org/linux/man-pages/man3/mkdtemp.3.html
  // RETURN VALUE
  // mkdtemp() function returns a pointer to the modified template string on
  // success, and NULL on failure, in which case errno is set appropriately.
  std::string modified_template {::mkdtemp(template_string.data())};

  return modified_template;
}

} // namespace Tools
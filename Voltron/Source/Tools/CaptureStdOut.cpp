#include "CaptureStdOut.h"

#include <cstdio>
#include <fstream>
#include <string>
#include <unistd.h>

namespace Tools
{

CaptureStdoutFixture::CaptureStdoutFixture():
  temp_file_{"/tmp/capture_stdout_XXXXXX"},
  restored_{false}
{
  // Create temporary file
  char* temp {const_cast<char*>(temp_file_.c_str())};
  int fd {::mkstemp(temp)};
  temp_file_ = std::string(temp);
  ::close(fd);
  
  // Save original stdout
  original_stdout_fd_ = ::dup(STDOUT_FILENO);
  
  // Redirect stdout to temp file
  ::freopen(temp_file_.c_str(), "w", stdout);
}  

CaptureStdoutFixture::~CaptureStdoutFixture()
{
  restore_stdout();
  // Clean up temp file
  if (!temp_file_.empty())
  {
    ::unlink(temp_file_.c_str());
  }
}

void CaptureStdoutFixture::capture_stdout()
{
  // Flush stdout to ensure data is written
  ::fflush(stdout);

  // Read captured output from temp file WITHOUT restoring stdout
  std::ifstream file(temp_file_);
  if (file)
  {
    local_oss_.str("");  // Clear previous content
    local_oss_ << file.rdbuf();
    file.close();
  }  

  // TRUNCATE THE FILE after reading so next capture starts fresh
  std::ofstream truncate_file(temp_file_, std::ios::trunc);
  truncate_file.close();

  // REOPEN stdout to the truncated file
  ::freopen(temp_file_.c_str(), "w", stdout);
}

void CaptureStdoutFixture::restore_stdout()
{
  if (restored_)
  {
    return;
  }

  // Flush stdout
  ::fflush(stdout);
  
  // Restore original stdout
  ::dup2(original_stdout_fd_, STDOUT_FILENO);
  ::close(original_stdout_fd_);
  
  // Read captured output from temp file
  std::ifstream file(temp_file_);
  if (file)
  {
    local_oss_ << file.rdbuf();
    file.close();
  }
}

} // namespace Tools
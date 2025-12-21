// CaptureStdout.h
#ifndef TOOLS_CAPTURE_STDOUT_H
#define TOOLS_CAPTURE_STDOUT_H

#include <sstream>
#include <string>
#include <unistd.h>

namespace Tools
{

class CaptureStdoutFixture
{
  public:
    std::ostringstream local_oss_;
    std::string temp_file_;
    int original_stdout_fd_;

    CaptureStdoutFixture();

    virtual ~CaptureStdoutFixture();

    void capture_stdout();

    void restore_stdout();

  private:
    // Check against running restore_stdout twice.
    bool restored_;
};

} // namespace Tools

#endif // TOOLS_CAPTURE_STDOUT_H
//------------------------------------------------------------------------------
/// \file CaptureCout.h
/// \brief Help capture std::cout standard output.
///-----------------------------------------------------------------------------
#ifndef TOOLS_CAPTURE_COUT_H
#define TOOLS_CAPTURE_COUT_H

#include <functional> // std::reference_wrapper
#include <streambuf> // std::streambuf
#include <sstream> // std::ostringstream
#include <utility> // std::pair

namespace Tools
{

std::pair<std::reference_wrapper<std::ostringstream>, std::streambuf*>
  capture_cout();

std::streambuf* capture_cout(std::ostringstream& local_oss);

class CaptureCoutFixture
{
  public:

    std::ostringstream local_oss_;

    CaptureCoutFixture();

    ~CaptureCoutFixture();

  protected:

    void restore_cout();

    void capture_locally();

  private:

    std::streambuf* cout_buffer_ptr_; // Save previous buffer.
};

} // namespace Tools

#endif // TOOLS_CAPTURE_COUT_H
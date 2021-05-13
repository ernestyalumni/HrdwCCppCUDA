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

class CaptureCout
{
  public:

    CaptureCout();

    ~CaptureCout();

    void capture_locally();

    //std::ostringstream& operator()();

    void restore_cout();

  private:

    std::ostringstream oss_;
    std::streambuf* cout_buffer_ptr_;
};

class CaptureCoutFixture
{
  public:

    std::ostringstream local_oss_;

    std::streambuf* cout_buffer_ptr_; // Save previous buffer.

    CaptureCoutFixture();

    void restore_cout();

  protected:

    void capture_locally();
};

} // namespace Tools

#endif // TOOLS_CAPTURE_COUT_H
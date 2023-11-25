//------------------------------------------------------------------------------
/// \brief Help capture std::cout standard output.
///-----------------------------------------------------------------------------
#ifndef UTILITIES_TESTING_CAPTURE_COUT_H
#define UTILITIES_TESTING_CAPTURE_COUT_H

#include <functional> // std::reference_wrapper
#include <streambuf> // std::streambuf
#include <sstream> // std::ostringstream
#include <utility> // std::pair

namespace Utilities
{
namespace Testing
{

//------------------------------------------------------------------------------
/// \return The original stream buffer that we've displaced, so that it can be
/// used again to restore the std::cout buffer.
//------------------------------------------------------------------------------
std::streambuf* capture_cout(std::ostringstream& local_oss);

class CaptureCout
{
  public:

    // Buffer to capture cout; it essentially displaces the stream of std::cout.
    std::ostringstream local_oss_;

    CaptureCout();

    ~CaptureCout();

  protected:

    void restore_cout();

    void capture_locally();

  private:

    // Save previous buffer.  
    std::streambuf* cout_buffer_ptr_;
};

} // namespace Testing
} // namespace Utilities

#endif // UTILITIES_TESTING_CAPTURE_COUT_H
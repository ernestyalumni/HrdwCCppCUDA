//------------------------------------------------------------------------------
/// \file CheckReturns.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Helper functions to check file descriptors.
/// \ref https://linux.die.net/man/3/clock_gettime     
/// \details 
/// \copyright If you find this code useful, feel free to donate directly and
/// easily at this direct PayPal link: 
///
/// https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
/// 
/// which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.  
/// Otherwise, I receive emails and messages on how all my (free) material on
/// physics, math, and engineering have helped students with their studies, and
/// I know what it's like to not have money as a student, but love physics (or 
/// math, sciences, etc.), so I am committed to keeping all my material 
/// open-source and free, whether or not sufficiently crowdfunded, under the 
/// open-source MIT license: feel free to copy, edit, paste, make your own
/// versions, share, use as you wish.    
/// Peace out, never give up! -EY
//------------------------------------------------------------------------------
/// COMPILATION TIPS:
///  g++ -std=c++14 CheckReturns_main.cpp -o CheckReturns_main
//------------------------------------------------------------------------------
#ifndef _UTILITIES_CHECK_RETURNS_H_
#define _UTILITIES_CHECK_RETURNS_H_

#include <cstring> // strerror
#include <iostream>
#include <stdexcept> // std::runtime_error
#include <string>
#include <system_error> 

namespace Utilities
{

//------------------------------------------------------------------------------
/// \brief Virtual base class for C++ functor for checking the result of system
/// calls
//------------------------------------------------------------------------------
class CheckReturn
{
  public:

    CheckReturn() = default;

    virtual int operator()(int result, const std::string& custom_error_string)
    {
      if (result < 0)
      {
        std::cout << " errno : " << std::strerror(errno) << '\n';
        throw std::system_error(
          errno,
          std::generic_category(),
          "Failed to " + custom_error_string + "\n");
        return errno;
      } 
      else
      {
        return result;
      }      
    }

    virtual int operator()(int result)
    {
      return this->operator()(
        result,
        "Integer return value to check was less than 0.");
    }
};

//------------------------------------------------------------------------------
// \ref http://man7.org/linux/man-pages/man2/open.2.html
// \details On error, -1 is returned, and errno is set appropriately.
// Otherwise, return new fd.
//------------------------------------------------------------------------------
class CheckOpen : public CheckReturn
{
  public:

    CheckOpen() = default;

    int operator()(int result)
    {
      return this->operator()(
        result,
        "Failed to open fd (::open())");
    }

  private:

    using Utilities::CheckReturn::operator();
};

//------------------------------------------------------------------------------
// \ref http://man7.org/linux/man-pages/man2/close.2.html
// \details On error, -1 is returned, and errno is set appropriately.
// 0 indicates success.
//------------------------------------------------------------------------------
class CheckClose : public CheckReturn
{
  public:

    CheckClose() = default;

    int operator()(int result)
    {
      return this->operator()(
        result,
        "Failed to close fd (::close())");
    }

  private:

    using Utilities::CheckReturn::operator();
};

//------------------------------------------------------------------------------
// \details Specified by Linux manual page, in that system calls that include
// - ::clock_gettime
// - ::clock_settime
// - ::clock_getres
// are documented to return 0 for success, or -1 for failure (in which case
// errno is set appropriately), and that errno will be returned by this
// function.
//------------------------------------------------------------------------------
int check_valid_fd(int e, const std::string& custom_error_string)
{
  if (e < 0)
  {
    std::cout << " errno : " << std::strerror(errno) << '\n';
    throw std::system_error(
      errno,
      std::generic_category(),
      "Failed to " + custom_error_string + "\n");
    return errno;
  } 
  else
  {
    return e;
  }
}

//------------------------------------------------------------------------------
// \ref https://linux.die.net/man/2/read
// \details On error, -1 is returned, and errno is set appropriately. In this
// case it's left unspecified whether file position (if any) changes.
// It's not an error if number of bytes read is smaller than number of bytes
// requested; this may happen for example because fewer bytes are actually
// available right now (maybe because we were close to end-of-file, or because
// we're reading from a pipe, or terminal.)
//------------------------------------------------------------------------------
template <typename T>
void check_read(const ssize_t number_of_bytes)
{
  if (number_of_bytes < 0)
  {
    std::cout << " errno : " << std::strerror(errno) << '\n';
    throw std::system_error(
      errno,
      std::generic_category(),
      "Failed to read (::read)\n");
  }
  else if (number_of_bytes != sizeof(T))
  {
    throw std::runtime_error(
      "number of bytes read is smaller than number of bytes requested.");
  }
}

//------------------------------------------------------------------------------
// \ref https://linux.die.net/man/2/write
// \details On error, -1 is returned, and errno is set appropriately.
// 0 indicates nothing was written.
// On success, number of bytes written is returned.
//------------------------------------------------------------------------------
template <ssize_t ExpectedNumberOfBytes>
void check_write(const ssize_t number_of_bytes)
{
  if (number_of_bytes < 0)
  {
    std::cout << " errno : " << std::strerror(errno) << '\n';
    throw std::system_error(
      errno,
      std::generic_category(),
      "Failed to write (::write)\n");
  }
  else if (number_of_bytes == 0)
  {
    throw std::runtime_error("Nothing was written.");
  }
  else if (number_of_bytes != ExpectedNumberOfBytes)
  {
    throw std::runtime_error("Number of bytes written was not expected.");
  }
}

//------------------------------------------------------------------------------
// \ref http://man7.org/linux/man-pages/man2/close.2.html
// \details On error, -1 is returned, and errno is set appropriately.
// 0 indicates success.
//------------------------------------------------------------------------------
int check_close(int e)
{
  if (e < 0)
  {
    std::cout << " errno : " << std::strerror(errno) << '\n';
    throw std::system_error(
      errno,
      std::generic_category(),
      "Failed to close fd (::close())");
  }
  return errno;
}


} // namespace Utilities

#endif // _UTILITIES_CHECK_FDS_H_

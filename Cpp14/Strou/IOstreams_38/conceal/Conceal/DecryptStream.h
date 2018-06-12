//------------------------------------------------------------------------------
/// \file DecryptStream.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  istream examples
/// \ref https://github.com/facebook/conceal/blob/master/native/cpp/DecryptStream.h    
/// \details istream examples. 
/// \copyright If you find this code useful, feel free to donate directly and easily at 
/// this direct PayPal link: 
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
///  g++ -std=c++14 istream_eg.cpp -o istream_eg
//------------------------------------------------------------------------------
#ifndef _DECRYPT_STREAM_H_
#define _DECRYPT_STREAM_H_

#include <istream>
#include <memory>
#include <streambuf>

namespace Conceal
{

class DecryptBuffer
{
  public:
};

struct BaseDecryptStream
{
  std::unique_ptr<std::istream> stream_;

  BaseDecryptStream(std::unique_ptr<std::istream> source);
};

class DecryptStream: private BaseDecryptStream, public std::istream
{
  public:
    DecryptStream(std::unique_ptr<std::istream> source);
    bool start();
    // Streams don't seem to play well with exceptions ... so let's inform of 
    // tag validity with a bool.
    bool validTag();
};



} // namespace Conceal

#endif // _DECRYPT_STREAM_H
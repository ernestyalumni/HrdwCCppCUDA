//------------------------------------------------------------------------------
/// \file CopyOnWrite.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Copy-on-write (COW) idiom.  
/// \ref 17.5.1.3 The Meaning of Copy Ch. 17 Templates; Bjarne Stroustrup, 
///   The C++ Programming Language, 4th Ed., Stroustrup; Ch. 17 Construction
/// https://en.wikibooks.org/wiki/More_C%2B%2B_Idioms/Copy-on-write
/// \details Idea is that a copy doesn't actually need independence until a 
///   shared state is written to, so we can delay the copying of the shared
///    state until just before the first write to it.
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
///  nvcc -std=c++14 Array_main.cpp -o Array_main
//------------------------------------------------------------------------------
#ifndef _COPY_ON_WRITE_H_
#define _COPY_ON_WRITE_H_

#include <memory> // std::shared_ptr

namespace Idioms
{

template <class T>
class CopyOnWrite
{
  public:

    explicit CopyOnWrite(T* t):
      sp_{t}
    {}

    const T& operator*() const
    {
      return *sp_;
    }

    T& operator&()
    {
      detach();
      return *sp_;
    }

    const T* operator->() const
    {
      return sp_.operator->();
    }

    T* operator->()
    {
      detach();
      return sp_.operator->();
    }

  private:

    void detach()
    {
      T* tmp = sp_.get();
      if (!(tmp == 0 || sp_.unique()))
      {
//        sp_ = std::make_shared<T>(tmp);
        sp_ = std::shared_ptr<T>(new T(*tmp));
      }
    }

    std::shared_ptr<T> sp_;
};

} // namespace Idioms

#endif // _COPY_ON_WRITE_H_

//------------------------------------------------------------------------------
/// \file Endian.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  Utility functions and classes that depend on bytesex, i.e.Endianness.
/// \ref https://github.com/google/sensei/blob/master/sensei/util/endian.h
/// \details std::array. 
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
///  g++ -std=c++14 stdarray.cpp -o stdarray
//------------------------------------------------------------------------------
#ifndef ENDIAN_ENDIAN_H
#define ENDIAN_ENDIAN_H

#include <cassert>

//------------------------------------------------------------------------------
/// \brief Define ValueType->IntType mapping for the unified 
///   "IntType FromHost (ValueType)" API. The mapping is implemented via"




//------------------------------------------------------------------------------
/// \brief Utilities to convert numbers between current host's native byte
///   order and big-endian byte order (same as network byte order)
/// \details Load/Store methods are alignment safe
//------------------------------------------------------------------------------
class BigEndian
{
  public:

    //--------------------------------------------------------------------------
    /// \name Unified BigEndian::Load/Store<T> API
    /// \brief Returns the T value encoded by the leading bytes of 'p',
    ///   interpreted according to the format specified below. 'p' has no 
    ///   alignment restrictions.
    ///
    /// Type            Format
    /// ------------    --------------------------------------------------------
    /// float, double   Big-endian IEEE-754 format.
    //--------------------------------------------------------------------------
    template<typename T>
    static T Load(const char* p);

    //--------------------------------------------------------------------------
    /// \brief Encodes 'value' in the format corresponding to T. Supported 
    ///   types are described in Load<T>(). 'p' has no alignment restrictions. 
    ///   In-place Store is safe (that is, it is safe to call
    ///     Store(x, reinterpret_cast<char*>(&x)).
    //--------------------------------------------------------------------------
    template <typename T>
    static void Store(T value, char* p);

}; // BigEndian




#endif // ENDIAN_ENDIAN_H

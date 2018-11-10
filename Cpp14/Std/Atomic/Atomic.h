//------------------------------------------------------------------------------
/// \file Atomic.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  std::atomic examples
/// \ref https://en.cppreference.com/w/cpp/atomic/atomic
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
///
/// \details Partial specializations (cf. cppreference); std library provides
/// partial specializations of `std::atomic` template for following types with
/// additional properties that primary template doesn't have:
/// 2) for pointer types, `std::atomic<T*>`, with `fetch_add`, `fetch_sub`.
//------------------------------------------------------------------------------
/// COMPILATION TIPS:
///  g++ -std=c++14 Atomic_main.cpp -o Atomic_main
//------------------------------------------------------------------------------

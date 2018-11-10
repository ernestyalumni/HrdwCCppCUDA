//------------------------------------------------------------------------------
/// \file Queue.h
/// \author Ernest Yeung
/// \email  ernestyalumni@gmail.com
/// \brief  std::list examples
/// \ref https://en.cppreference.com/w/cpp/container/list 
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
/// \details std::list implementations.
/// std::list is a container that supports constant time insertion and removal
/// of elements from anywhere in the container.
/// Fast random access isn't supported.
/// Usually implemented as doubly-linked list.
/// Compared to std::forward_list this container provides bidirectional
/// iteration capability while being less space efficient.
/// Adding, removing, and moving elements within list or across several lists
/// doesn't invalidate iterators or references. Iterator is invalidated only
/// when corresponding element is deleted.
//------------------------------------------------------------------------------
/// COMPILATION TIPS:
///  g++ -std=c++14 Queue_main.cpp -o Queue_main
//------------------------------------------------------------------------------

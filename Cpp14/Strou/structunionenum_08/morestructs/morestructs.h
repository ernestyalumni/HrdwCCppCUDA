/**
 * @file   : morestructs.h
 * @brief  : morestructs in header file, in C++14, 
 * @details : struct with smart ptrs, unique ptrs
 * 
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20171229    
 * @ref    : Ch. 8 Structures, Unions, and Enumerations; Bjarne Stroustrup, The C++ Programming Language, 4th Ed.  
 * Addison-Wesley 
 * 
 * https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
 * 
 * which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.  
 * Otherwise, I receive emails and messages on how all my (free) material on 
 * physics, math, and engineering have helped students with their studies, 
 * and I know what it's like to not have money as a student, but love physics 
 * (or math, sciences, etc.), so I am committed to keeping all my material 
 * open-source and free, whether or not 
 * sufficiently crowdfunded, under the open-source MIT license: 
 * 	feel free to copy, edit, paste, make your own versions, share, use as you wish.  
 *  Just don't be an asshole and not give credit where credit is due.  
 * Peace out, never give up! -EY
 * 
 * */
/* 
 * COMPILATION TIP
 * g++ main.cpp ./structs/structs.cpp -o main
 * 
 * */
#ifndef __MORESTRUCTS_H__
#define __MORESTRUCTS_H__

#include <memory>  // std::unique_ptr std::make_unique  
#include <array>   // std::array

struct unique_S {
	// (data) members
	std::unique_ptr<int[]> S_uptr;   
	
};

struct unique_S1 {
	// (data) members
	// data
	std::unique_ptr<int[]> S_uptr;   
	// dimensions describing the data
	std::array<size_t,2> L_is; // L_i = (L_x,L_y); use size_t instead of int because size_t is "bigger"
	size_t L; // L = L_x*L_y 
	
	// constructors 
	unique_S1() = default; // default constructor 
	unique_S1(std::array<size_t,2> & L_is); 
};


#endif // END of __MORESTRUCTS_H__

/**
 * @file   : main_morestructs.cpp
 * @brief  : main driver file for more structs in C++11/14, 
 * @details : more structs with smart ptrs, unique_ptrs
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
 * g++ main_morestructs.cpp ./morestructs/morestructs.cpp -o main_morestructs
 * 
 * */
#include "./morestructs/morestructs.h"  // unique_S, unique_S1

#include <iostream> 

int main(int argc, char* argv[]) {
	unique_S unique_s_inst;  
	
	std::cout << " sizeof(unique_S) : " << sizeof(unique_s_inst) << std::endl; 

	unique_S1 unique_s1_inst; 
	std::cout << " sizeof(unique_S1) : " << sizeof(unique_s1_inst) << std::endl; 

	std::array<size_t,2> L_is2 { 256, 128 }; 
	
	unique_S1 unique_s1_inst1 = { L_is2 }; 

	std::cout << unique_s1_inst1.L_is[0] << " " << unique_s1_inst1.L_is[1] <<  " " << 
		unique_s1_inst1.L << std::endl; 
		

}


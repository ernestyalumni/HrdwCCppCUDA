/**
 * @file   : rvalues.cpp
 * @brief  : rvalue reference in C++11, examples     
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20180125
 * @ref    : http://www.artima.com/cppsource/rvalue.html
 * @details : 
 * 
 * If you find this code useful, feel free to donate directly and easily at this direct PayPal link: 
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
 * g++ -std=c++14 -pthread thread2.cpp -o thread2
 * 
 * */
#include <iostream>
#include <string>
#include <vector>
#include <memory> // std::unique_ptr, 
#include <algorithm> // std::fill

/**
 * @ref http://www.artima.com/cppsource/rvalue.html
 * Perfect Forwarding
 */
template <class T>
std::shared_ptr<T>
factory() // no argument version
{
	return std::shared_ptr<T>(new T);
}

template <class T, class A1>
std::shared_ptr<T>
factory(const A1& a1) // 1 argument version
{
	return std::shared_ptr<T>(new T(a1));
}

// all the other versions



int main(int argc, char* argv[]) 
{
	// "boilerplate" test values 

	constexpr const unsigned int N_fvec = 32;
	std::vector<float> fvec(N_fvec);
	std::fill( fvec.begin(), fvec.end(), 1.f);  

	std::string test_str = "Hello world."; 

	auto test_uniqptr2farr = std::make_unique<float[]>( N_fvec);
	for (auto i=0; i< N_fvec; i++) {
		test_uniqptr2farr[i] = static_cast<float>(i) + 2.5f;
	}
	
	// END of "boilerplate" test values

		// lvalue references
	std::vector<float>& fvec_ref1 = fvec; // an lvalue reference
	std::string& test_str_ref1 = test_str; 
	std::unique_ptr<float[]>& test_uniqptr2farr_ref1 = test_uniqptr2farr;  
	
	for (auto i=0; i< N_fvec; i++) {
		std::cout << test_uniqptr2farr[i] << " " ; 
	}
	std::cout << std::endl; 

	for (auto i=0; i< N_fvec; i++) {
		std::cout << test_uniqptr2farr_ref1[i] << " " ; 
	}
	std::cout << std::endl; 
	// so last 2 print outs show you can copy the entire contents of unique_ptr to 
	// a newly created unique_ptr object 	

	// rvalue references
//	std::vector<float>&& fvec_ref2 = fvec; //  error: cannot bind ‘std::vector<float>’ lvalue to ‘std::vector<float>&&’
	std::vector<float>&& fvec_ref2 = std::move( fvec); 

//	std::vector<float>&& fvec_ref2  = std::move( fvec ); // an rvalue reference

//	std::string&& test_str_ref2 = test_str; // error: cannot bind lvalue to &&
	std::string&& test_str_ref2 = "Hello hello world.";

	std::unique_ptr<float[]>&& test_uniqptr2farr_ref2 = std::move(test_uniqptr2farr);  // error: cannot bind lvalue to &&

	// sanity check
	for (auto i=0; i< N_fvec; i++) {
		std::cout << fvec[i] << " " ; 
	}
	std::cout << std::endl; 

	for (auto i=0; i< N_fvec; i++) {
		std::cout << fvec_ref2[i] << " " ; 
	}
	std::cout << std::endl; 
	
	for (auto i=0; i< N_fvec; i++) {
		std::cout << test_uniqptr2farr[i] << " " ; 
	}
	std::cout << std::endl; 

	for (auto i=0; i< N_fvec; i++) {
		std::cout << test_uniqptr2farr_ref2[i] << " " ; 
	}
	std::cout << std::endl; 
	
	// Move Semantics 
	// Eliminating spurious copies  

	auto fvec3 = std::move(fvec); 
	// sanity check
	/* Segmentation fault
	for (auto i=0; i< N_fvec; i++) {
		std::cout << fvec[i] << " " ; 
	}
	* */
	std::cout << std::endl; 
	std::cout << std::endl << " fvec3 : " << std::endl;
	for (auto i=0; i< N_fvec; i++) {
		std::cout << fvec3[i] << " " ; 
	}
	std::cout << std::endl; 


//	fvec3 = std::move(fvec_ref2); // Segmentation Fault, when you move in a 2nd. time for a guy
	/* Segmentation Fault
	for (auto i=0; i< N_fvec; i++) {
		std::cout << fvec_ref2[i] << " " ; 
	}
	*/ 
/*	std::cout << std::endl << " fvec3 : " << std::endl;
	for (auto i=0; i< N_fvec; i++) {
		std::cout << fvec3[i] << " " ; 
	}
	std::cout << std::endl; 
*/
//	std::cout << fvec3[0] << std::endl; // Segmentation Fault

	std::cout << test_str << std::endl; 
	test_str_ref1 = std::move(test_str) ; 
	std::cout << test_str_ref1 << std::endl; 
	std::cout << test_str << std::endl; 
	
	
}

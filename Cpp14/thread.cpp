/**
 * @file   : thread.cpp
 * @brief  : thread library C/C++11, examples   
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20180124
 * @ref    : http://en.cppreference.com/w/cpp/thread/thread 
 * https://stackoverflow.com/questions/34933042/undefined-reference-to-pthread-create-error-when-making-c11-application-with
 * @details : class thread represents *a single thread of execution*.  
 * Threads allow multiple functions to execute concurrently.  
 * 
 * "The fact that a program is written in C++11 has no bearing on with whether or not it needs to be linked with the pthread library. It needs to link that library if it requires Posix threads.
 * 
 * C++11 provides the std::thread class and each conforming compiler's standard library must implement the functionality of that class using some native threads API hosted by the target system. GCC implements it using pthreads, so you cannot any build a program that creates std::thread objects with GCC unless you link it with -pthread"
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
 * g++ -std=c++14 -pthread thread.cpp -o thread
 * 
 * */
// @ref cf. http://www.cplusplus.com/reference/thread/thread/

// thread example 
#include <iostream> 		// std::cout
#include <thread>  			// std::thread

void foo()
{
	// do stuff ...
}

void bar(int x)
{
	// do stuff ...
}

int main(int argc, char* argv[]) 
{
	std::thread first (foo); 		// spawn new thread that calls foo()
	std::thread second (bar,0); 	// spawn new thread that calls bar(0)
	
	std::cout << "main, foo and bar now execute concurrently...\n";
	
	// synchronize threads:
	first.join();					// pauses until first finishes
	second.join();					// pauses until second finishes
	
	std::cout << "foo and bar completed.\n";
	
	return 0;


}


	
	

 

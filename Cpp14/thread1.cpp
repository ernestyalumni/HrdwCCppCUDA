/**
 * @file   : thread1.cpp
 * @brief  : thread library C/C++11, examples   
 * @author : Ernest Yeung <ernestyalumni@gmail.com>
 * @date   : 20180124
 * @ref    : http://en.cppreference.com/w/cpp/thread/thread 
 * https://solarianprogrammer.com/2011/12/16/cpp-11-thread-tutorial/
 * https://stackoverflow.com/questions/34933042/undefined-reference-to-pthread-create-error-when-making-c11-application-with
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
 * g++ -std=c++14 -pthread thread1.cpp -o thread1
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

// This function will be called from a thread
void call_from_thread() {
	std::cout << "Hello, World" << std::endl; 
}

static const int num_threads = 10; 

int main(int argc, char* argv[]) 
{
	std::thread first (foo); 		// spawn new thread that calls foo()
	std::thread second (bar,0); 	// spawn new thread that calls bar(0)
	
	std::cout << "main, foo and bar now execute concurrently...\n";
	
	// synchronize threads:
	first.join();					// pauses until first finishes
	second.join();					// pauses until second finishes
	
	std::cout << "foo and bar completed.\n";
	
	// Launch a thread 
	std::thread t1(call_from_thread); 
	
	// Join the thread with the main thread
	t1.join(); // pauses until t1 finishes
	
	/* 
	 * Usually we will want to launch more than one thread at once and 
	 * do some work in parallel. 
	 * In order to do this we could create an array of threads versus 
	 * creating a single thread like in our first example. 
	 * In the next example the main function creates a group of 10 threads 
	 * that will do some work and waits for the threads to finish their work 
	 * */
	
	std::thread t[num_threads];
	
	// Launch a group of threads
	for (int i=0; i< num_threads; ++i) {
		t[i] = std::thread(call_from_thread);
	}
	
	std::cout << "Launched from the main\n"; 
	
	// Join the threads with the main thread
	for (int i=0; i < num_threads; ++i) {
		t[i].join();
	}
	
	return 0;
	
	
}

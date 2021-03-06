/**
 * @file   : Thread.h
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : Thread RAII (Resource Acquisition Is Initialization)
 * @ref    : https://stackoverflow.com/questions/35150629/stdthread-detachable-and-exception-safety
 * 
 * If you find this code useful, feel free to donate directly and easily at this direct PayPal link: 
 * 
 * https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
 * 
 * which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.  
 * Otherwise, I receive emails and messages on how all my (free) material on physics, math, and engineering have 
 * helped students with their studies, and I know what it's like to not have money as a student, but love physics 
 * (or math, sciences, etc.), so I am committed to keeping all my material open-source and free, whether or not 
 * sufficiently crowdfunded, under the open-source MIT license: 
 * 	feel free to copy, edit, paste, make your own versions, share, use as you wish.    
 * Peace out, never give up! -EY
 * 
 * COMPILATION TIPS:
 *  g++ -std=c++17 -c factor.cpp
 * */
#include <thread>

class BasicThread
{
	public:

		BasicThread();

		explicit BasicThread(std::thread&& th): 
			th_{std::move(th)}
		{}

		BasicThread(const BasicThread&) = delete;
		BasicThread(BasicThread&&) = default;
		BasicThread& operator=(const BasicThread&) = delete;
		BasicThread& operator=(BasicThread&&) = default;

		~BasicThread()
		{
			if (th_.joinable())
			{
				th_.join();
			}
		}

	private:
		std::thread th_;
};

template <class Executable>
class Thread
{
	public:

		template <typename ... Args>
		explicit Thread(Args&&... arguments):
			executable_{std::forward<Args>(arguments)...},
			th_{std::thread{executable_.run}}
		{}

		// Not copyable, but movable
		Thread(const Thread&) = delete;
		Thread& operator=(const Thread&) = delete;
		Thread(Thread&&) = default;
		Thread& operator=(Thread&&) = default;

		void run_it()
		{
			executable_.run();
		}

		~Thread()
		{
			if (th_.joinable())
			{
				th_.join();
			}
		}

	private:

		Executable executable_;
		std::thread th_;

};

/// \details Use CRTP (Curiously Recurring template pattern), idiom in C++ in
/// which class X derived from a class template instantiation using X itself
/// as template argument.
template <class T>
class ThreadRAII : public T
{
	public:

		template <typename ... Args>
		explicit ThreadRAII(Args&&... arguments):
			th_{}

};

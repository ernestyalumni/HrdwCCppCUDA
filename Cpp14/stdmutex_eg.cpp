/**
 * @file   : stdmutex_eg.cpp
 * @author : Ernest Yeung
 * @email  : ernestyalumni@gmail.com
 * @brief  : Example of std::mutex.  
 * @details mutex class is a synchronization primitive used to protect shared
 * data from being simultaneously accessed by multiple threads
 * @ref    : http://en.cppreference.com/w/cpp/utility/initializer_list
 * If you find this code useful, feel free to donate directly and easily at this direct PayPal link: 
 * 
 * https://www.paypal.com/cgi-bin/webscr?cmd=_donations&business=ernestsaveschristmas%2bpaypal%40gmail%2ecom&lc=US&item_name=ernestyalumni&currency_code=USD&bn=PP%2dDonationsBF%3abtn_donateCC_LG%2egif%3aNonHosted 
 * 
 * which won't go through a 3rd. party such as indiegogo, kickstarter, patreon.  
 * Otherwise, I receive emails and messages on how all my (free) material on physics, math, and engineering have 
 * helped students with their studies, and I know what it's like to not have money as a student, but love physics 
 * (or math, sciences, etc.), so I am committed to keeping all my material open-source and free, whether or not 
 * sufficiently crowdfunded, under the open-source MIT license: 
 *  feel free to copy, edit, paste, make your own versions, share, use as you wish.    
 * Peace out, never give up! -EY
 * 
 * COMPILATION TIPS:
 *  g++ -std=c++14 -lpthread stdmutex_eg.cpp -o stdmutex_eg
 * */
#include <iostream>
#include <map>
#include <string>
#include <chrono>
#include <thread>
#include <mutex>

//------------------------------------------------------------------------------
/// \details mutex class is a synchronization primitive that can be used to 
/// protect shared data from being simultaneously accessed by multiple threads.
/// mutex offers exclusive, non-recursive ownership semantics:
/// * calling thread owns mutex from time it successfully calls either lock or
/// try_lock until it calls unlock
/// * When thread owns mutex, all other threads will block (for calls to lock)
/// or receive a false return value (for try_lock) if they attempt to claim 
/// ownership of the mutex.
/// * A calling thread must not own the mutex prior to calling lock or try_lock
/// std::mutex is neither copyable nor movable.
/// 
/// Notes
/// std::mutex usually not accessed directly: std::unique_lock, std::lock_guard
/// manage locking in a more exception-safe manner.
//------------------------------------------------------------------------------

std::map<std::string, std::string> g_pages;
std::mutex g_pages_mutex;

void save_page(const std::string &url)
{
	// simulate a long page fetch
	std::this_thread::sleep_for(std::chrono::seconds(2));
	std::string result {"fake content"};

	std::lock_guard<std::mutex> guard(g_pages_mutex);
	g_pages[url] = result;
}

std::map<std::string, std::string> g_pages2;
std::mutex g_pages_mutex2;

void save_page2(const std::string& url)
{
	// simulate a long page fetch
	std::cout << " Sleep for 3 seconds\n ";
	std::this_thread::sleep_for(std::chrono::seconds(3));

	std::string result {"fake content 2"};

	std::cout << " Locking guard for g_pages_mutex2\n";
	std::lock_guard<std::mutex> guard(g_pages_mutex2);

	g_pages2[url] = result;
	std::cout << " Sleep for 1 seconds\n";
	std::this_thread::sleep_for(std::chrono::seconds(1));

}

class MutexContainer
{
	public:

		MutexContainer() = default;

		MutexContainer(const MutexContainer&) = delete;
		MutexContainer(MutexContainer&&) = delete;
		MutexContainer& operator=(const MutexContainer&) = delete;
		MutexContainer& operator=(MutexContainer&&) = delete;

		~MutexContainer() = default;

		void lockGuard()
		{
			std::lock_guard<std::mutex> guard(mutex_);
		}

	private:
		std::mutex mutex_;
};

class MutexContainer2
{
	public:

		MutexContainer2() = default;

		explicit MutexContainer2(std::mutex& mutex):
			mutex_{mutex}
		{}

		MutexContainer2(const MutexContainer2&) = delete;
		MutexContainer2(MutexContainer2&&) = delete;
		MutexContainer2& operator=(const MutexContainer2&) = delete;
		MutexContainer2& operator=(MutexContainer2&&) = delete;

		~MutexContainer2() = default;

		void lockGuard()
		{
			std::lock_guard<std::mutex> guard(mutex_);
		}

	private:
		std::mutex& mutex_;
};


MutexContainer mutex_container;
void save_page3(const std::string& url)
{
	// simulate a long page fetch
	std::cout << " Sleep for 4 seconds\n ";
	std::this_thread::sleep_for(std::chrono::seconds(4));

	std::string result {"fake content 3"};

	std::cout << " Locking guard for mutex_container \n";
	mutex_container.lockGuard();

	g_pages2[url] = result;
	std::cout << " Sleep for 5 seconds\n";
	std::this_thread::sleep_for(std::chrono::seconds(5));
}

void save_page4(const std::string& url, MutexContainer2& mutex_container)
{
	// simulate a long page fetch
	std::cout << " Sleep for 6 seconds\n ";
	std::this_thread::sleep_for(std::chrono::seconds(6));

	std::string result {"fake content 4"};

	std::cout << " Locking guard for mutex_container \n";
	mutex_container.lockGuard();

	g_pages2[url] = result;
	std::cout << " Sleep for 7 seconds\n";
	std::this_thread::sleep_for(std::chrono::seconds(7));
}


int main()
{
	std::thread t1(save_page, "http://foo");
	std::thread t2(save_page, "http://bar");
	t1.join();
	t2.join();

	// safe to access g_pages without lock now, as the threadas are joined
	for (const auto &pair : g_pages)
	{
		std::cout << pair.first << " -> " << pair.second << '\n';
	}

	std::thread t12(save_page2, "http://foo2");
	std::thread t22(save_page2, "http://bar2");
	t12.join();
	t22.join();

	// safe to access g_pages without lock now, as the threadas are joined
	for (const auto &pair : g_pages2)
	{
		std::cout << pair.first << " -> " << pair.second << '\n';
	}

	std::thread t13(save_page3, "http://foo3");
	std::thread t23(save_page3, "http://bar3");

	std::cout << " t13 is joinable? " << t13.joinable() << '\n';
	t13.join();
	std::cout << " t13 is joinable? " << t13.joinable() << '\n';
	t23.join();

	// safe to access g_pages without lock now, as the threadas are joined
	for (const auto &pair : g_pages2)
	{
		std::cout << pair.first << " -> " << pair.second << '\n';
	}

	MutexContainer2 mutex_container2{g_pages_mutex2};

/* in the following:
error: no type named ‘type’ in ‘class std::result_of<void (*(const char*, MutexContainer2))(const std::__cxx11::basic_string<char>&, MutexContainer2&)>’
       typedef typename result_of<_Callable(_Args...)>::type result_type;
*/
/*	std::thread t14(save_page4, "http://foo4", mutex_container2);
	t14.join();

	std::thread t24(save_page4, "http://bar4", mutex_container);
	t24.join();
*/


}

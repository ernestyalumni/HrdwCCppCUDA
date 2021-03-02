//------------------------------------------------------------------------------
/// \brief Demonstrate <csignal> standard library header.
/// \details Header <csignal> was originally C standard library <signal.h>
/// \ref https://en.cppreference.com/w/cpp/header/csignal
//------------------------------------------------------------------------------
#ifndef CPP_UTILITIES_SIGNAL_HANDLER_H
#define CPP_UTILITIES_SIGNAL_HANDLER_H

#include <csignal>

#include <functional>

namespace Cpp
{
namespace Utilities
{

struct SignalStatus 
{
	volatile std::sig_atomic_t atomic_signal_status_;
	int integer_signal_status_;	
};

auto create_signal_handler(SignalStatus& signal_status)
{
	/*
	return [&signal_status](int signal) -> void
	{
		signal_status.atomic_signal_status_ = signal;
		signal_status.integer_signal_status_ = signal;
	};*/

	class SignalHandler
	{
		public:
			SignalHandler(SignalStatus& signal_status):
				signal_status_{signal_status}
			{}

			void operator()(int signal)
			{
				signal_status_.atomic_signal_status_ = signal;
				signal_status_.integer_signal_status_ = signal;
			}

		private:

			SignalStatus& signal_status_;
	};

	return SignalHandler{signal_status};
}

struct InterruptSignalHandler
{
	InterruptSignalHandler(
		void (*signal_handler)(int),
		const int signal_type = SIGINT
		):
		original_signal_type_{signal_type},
		stored_signal_handler_(std::signal(signal_type, signal_handler))
	{}

	template <typename F>
	InterruptSignalHandler(
		F signal_handler,
		const int signal_type = SIGINT
		):
		original_signal_type_{signal_type},
		stored_signal_handler_(std::signal(signal_type, signal_handler))
	{}


	// Keep reusing same signal handler.
	//~InterruptSignalHandler()
	//{
	//	std::signal(original_signal_type_, stored_signal_handler_);
	//}

	/*
	void signal_handler(int signal)
	{
		atomic_signal_status_ = signal;
		integer_signal_status_ = signal;
	}
	*/

	int original_signal_type_;

	// TODO: Look up return value for std::signal.
	::sighandler_t stored_signal_handler_;
};

} // namespace Utilities
} // namespace Cpp

#endif // CPP_UTILITIES_SIGNAL_HANDLER_H
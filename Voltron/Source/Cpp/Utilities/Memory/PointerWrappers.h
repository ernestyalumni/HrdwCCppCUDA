//------------------------------------------------------------------------------
/// \file WrapperPointers.h
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating Pointers under encapsulation.
/// @ref https://ece.uwaterloo.ca/~dwharder/aads/Lecture_materials/
///-----------------------------------------------------------------------------
#ifndef CPP_UTILITIES_MEMORY_POINTER_WRAPPERS_H
#define CPP_UTILITIES_MEMORY_POINTER_WRAPPERS_H

#include <cassert>
#include <optional>
#include <memory>
#include <type_traits>
#include <utility> // for std::forward()

namespace Cpp
{
namespace Utilities
{
namespace Memory
{

template <typename T>
class WrappedUniquePtr
{
	public:

		struct ConstructNullPtr
		{};

		template <typename... Args>
		using unique_ptr = std::unique_ptr<Args...>;

		WrappedUniquePtr(unique_ptr<T> t):
			t_{std::move(t)}
		{}

		template <
			typename... Args,
			typename = std::enable_if_t<
				sizeof...(Args) != 0 ||
				std::is_default_constructible<T>::value
			>
		>
		WrappedUniquePtr(Args&&... args):
			t_{std::make_unique<T>(std::forward<Args>(args)...)}
		{}

		// TODO implement the reference constructor

		explicit WrappedUniquePtr(T&& input):
			t_{std::make_unique<T>(std::forward<T>(input))}
		{}

		// This is necessary in order to do this:

		explicit WrappedUniquePtr(ConstructNullPtr):
			t_{nullptr}
		{}

		//--------------------------------------------------------------------------
		// Non-copyable, non-movable (until needed).
		//--------------------------------------------------------------------------
		WrappedUniquePtr(const WrappedUniquePtr&) = delete;
		WrappedUniquePtr(WrappedUniquePtr&&) = delete;
		WrappedUniquePtr& operator=(const WrappedUniquePtr&) = delete;
		WrappedUniquePtr& operator=(WrappedUniquePtr&&) = delete;

		//--------------------------------------------------------------------------
		/// \brief Closes handle.
		//--------------------------------------------------------------------------
		virtual ~WrappedUniquePtr() = default;

		// deferences pointer to the managed object.
		T& get_object()
		{
			assert(t_);

			return *t_.get();
		}

		// cf. https://en.cppreference.com/w/cpp/memory/unique_ptr/operator_bool
		// Checks whether *this owns an object, i.e. whether get() != nullptr.
		bool owns_object() const
		{
			return static_cast<bool>(t_);
		}

		// cf. https://en.cppreference.com/w/cpp/memory/unique_ptr/release
		// Releases ownership of managed object if any. Caller is responsible for
		// deleting object.
		T* release()
		{
			return t_.release();
		}

		// cf. https://en.cppreference.com/w/cpp/memory/unique_ptr/reset
		// Replaces the managed object.
		// \param ptr - pointer to a new object to manager.
		void link_new(T* ptr)
		{
			t_.reset(ptr);
		}

	private:

		unique_ptr<T> t_;
};

} // namespace Memory
} // namespace Utilities
} // namesapce Cpp

#endif // CPP_UTILITIES_MEMORY_POINTER_WRAPPERS_H
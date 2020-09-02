//------------------------------------------------------------------------------
/// \file Lists.h
/// \author Ernest Yeung
/// \brief Classes and functions demonstrating lists as an Abstract Data
/// Structure.
/// @ref https://ece.uwaterloo.ca/~dwharder/aads/Lecture_materials/
///-----------------------------------------------------------------------------
#ifndef DATA_STRUCTURES_LISTS_LISTS_H
#define DATA_STRUCTURES_LISTS_LISTS_H

#include <memory>
#include <utility> // for std::forward()

namespace DataStructures
{

namespace Lists
{

namespace SinglyLinked
{

template <typename T>
class Node
{
	public:

		explicit Node(T&& value):
			value_{std::forward<T>(value)},
			next_{nullptr}
		{}

		// Without this constructor, lvalues cannot be inputted.
		explicit Node(T& value):
			value_{std::forward<T>(value)},
			next_{nullptr}
		{}


		Node(T&& value, std::unique_ptr<Node<T>> next):
			value_{std::forward<T>(value)},
			next_{std::move(next)}
		{}

		virtual ~Node() = default;

		T value() const
		{
			return value_;
		}

		// deferences pointer to the next object.
		Node<T>& get_next()
		{
			assert(next_);

			return *next_.get();
		}

		// Checks if next_ owns an object, i.e. whether get() != nullptr
		bool is_linked() const
		{
			return static_cast<bool>(next_);
		}

		// Releases ownership of managed object if any. Caller is responsible for
		// deleting object.
		Node<T>* release()
		{
			return next_.release();
		}

		void link_new(Node<T>* next)
		{
			next_.reset(next);
		}

	private:

		T value_;
		std::unique_ptr<Node<T>> next_;
};

} // namespace SinglyLinked

} // namespace Lists

} // namespace DataStructures


#endif // DATA_STRUCTURES_LISTS_LISTS_H
//------------------------------------------------------------------------------
/// \file StringViewAvoidsCopying.cpp
/// \author Ernest Yeung
/// \brief Demonstrate purpose of std::string_view is to avoid copying data
/// already owned by someone else and which only a non-mutating view is
/// required.
///
/// \details int main() is needed to overload new globally.
/// \ref https://www.modernescpp.com/index.php/c-17-avoid-copying-with-std-string-view
///-----------------------------------------------------------------------------

#include <cassert>
#include <cstdlib> // std::malloc
#include <iostream>
#include <string_view>

// Overload global operator new, to see which operation causes a memory
// allocation.
void* operator new(std::size_t count)
{
  std::cout << "   " << count << " bytes " << std::endl;

  // https://en.cppreference.com/w/cpp/memory/c/malloc
  // void* malloc(std::size_t size);
  // Allocates size bytes of uninitialized storage.
  // Return value: on success, returns pointer to beginning of newly
  // allocated memory. To avoid memory leak, returned pointed must be
  // deallocated with std::free() or std::realloc().
  return std::malloc(count);
}

void get_string(const std::string& str)
{}

void get_string_view(std::string_view str_view)
{}

int main()
{
  std::cout << std::endl;

  std::cout << "std::string" << std::endl;

  // Small string optimization.
  // strings stores data on heap, but only true if string exceeds an
  // implementation-dependent size.

  std::string small {"0123456789"};
  std::string substr {small.substr(5)};
  std::cout << "   " << substr << std::endl;  // 56789

  std::cout << std::endl;

  std::cout << "get_string" << std::endl;

  get_string(small);
  get_string("0123456789");
  const char message [] = "0123456789";
  get_string(message);

  std::cout << std::endl;

  {
    std::cout << "std::string" << std::endl;

    // 41 bytes
    std::string large {"0123456789-123456789-123456789-123456789"};
    // 31 bytes
    std::string substr {large.substr(10)};

    std::cout << std::endl;

    // Prove std::string_view allocates no memory, contrary to std::string.

    std::cout << "std::string_view" << std::endl;

    std::string_view large_string_view {large.c_str(), large.size()};
    large_string_view.remove_prefix(10);

    assert(substr == large_string_view);

    std::cout << std::endl;

    std::cout << "get_string" << std::endl;

    get_string(large);

    std::cout << "\n From string constant (r-value)\n";
    // 41 bytes
    get_string("0123456789-123456789-123456789-123456789");
    const char message [] = "0123456789-123456789-123456789-123456789";
    std::cout << "\n Construct from a char array\n";
    // 41 bytes
    get_string(message);

    std::cout << std::endl;

    std::cout << "get_string_view" << std::endl;

    get_string_view(large);
    get_string_view("0123456789-123456789-123456789-123456789");
    get_string_view(message);

    std::cout << std::endl;
  }

}
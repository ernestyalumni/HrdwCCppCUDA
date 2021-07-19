#ifndef UTILITIES_DEBUGGING_MACROS_H
#define UTILITIES_DEBUGGING_MACROS_H

//------------------------------------------------------------------------------
/// \ref https://stackoverflow.com/questions/15002550/c-macros-and-namespaces
/// \details Macros are handled by preprocessor, which knows nothing about
/// namespaces. So macros aren't namespaced, they're just text substitution.
/// Use of macros discouraged since thus they pollute global namespace.
//------------------------------------------------------------------------------

#ifndef NDEBUG

//------------------------------------------------------------------------------
/// \brief Library of debugging macros.  
//------------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib> // abort()

// IF PREDICATE is true, do nothing. Otherwise, print an error with the
// specified message to STDERR. This macros only operates when DEBUG = 1. This
// macro takes a PREDICATE to evaluate followed by the standard arguments to
// PRINTF().
#define DEBUG_ASSERT(PREDICATE, ...) \
  do { \
    if (!(PREDICATE)) { \
      fprintf(stderr, "%s:%d (%s) Assertion " #PREDICATE " failed: ", \
        __FILE__, __LINE__, __PRETTY_FUNCTION__); \
      fprintf(stderr, __VA_ARGS__); \
      abort(); \
    } \
  } while (0) \

// Case sensitive with macros.
#define debugging_assert DEBUG_ASSERT

#else

#define debugging_assert(...) // Do Nothing.

#endif

#endif // UTILITIES_DEBUGGING_MACROS_H

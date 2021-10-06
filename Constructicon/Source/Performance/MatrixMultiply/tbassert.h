#ifndef _TBASSERT_H_
#define _TBASSERT_H_ 1

#ifndef NDEBUG

/*******************************************************************************
 * Library of debugging macros.
 ******************************************************************************/

#include <stdio.h>
#include <stdlib.h>

// If PREDICATE is true, do nothing. Otherwise, print an error with the
// specified message to STDERR. This macro only operates when DEBUG = 1. This
// macros takes a PREDICATE to evaluate followed by the standard arguments to
// PRINTF().
#define DEBUG_ASSERT(PREDICATE, ...)                                  \
  do {                                                                \
    if (!(PREDICATE)) {                                               \
      fprintf(stderr, "%s:%d (%s) Assertion " #PREDICATE " failed: ", \
        __FILE__, __LINE__, __PRETTY_FUNCTION__);                      \
        fprintf(stderr, __VA_ARGS__);                                 \
        abort();                                                      \
      }                                                               \
    } while (0)


#define tbassert DEBUG_ASSERT

#else

#define tbassert(...) // Nothing.

#endif 

#endif // _TBASSERT_H_

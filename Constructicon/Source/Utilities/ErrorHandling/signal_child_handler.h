#ifndef UTILITIES_ERROR_HANDLING_SIGNAL_CHILD_HANDLER_H
#define UTILITIES_ERROR_HANDLING_SIGNAL_CHILD_HANDLER_H

namespace Utilities
{
namespace ErrorHandling
{

//------------------------------------------------------------------------------
/// pp. 938 Ch. 12 Concurrent Programming, 3rd. Ed. Computer Systems: A
/// Programmer's Perspective. Bryant and O'Hallaron.
//------------------------------------------------------------------------------
void signal_child_handler(const int signal);

} // namespace ErrorHandling
} // namespace Utilities

#endif // UTILITIES_ERROR_HANDLING_SIGNAL_CHILD_HANDLER_H

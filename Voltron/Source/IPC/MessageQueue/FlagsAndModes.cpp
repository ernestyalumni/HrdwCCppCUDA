#include "FlagsAndModes.h"

#include "Cpp/Utilities/TypeSupport/UnderlyingTypes.h"

using Cpp::Utilities::TypeSupport::get_underlying_value;

namespace IPC
{
namespace MessageQueue
{

int to_operation_flag_value(const OperationFlag operation_flag)
{
  return get_underlying_value<OperationFlag>(operation_flag);  
}

int to_additional_operation_flag_value(const AdditionalOperationFlag flag)
{
  return get_underlying_value<AdditionalOperationFlag>(flag);  
}

mode_t to_permission_mode_value(const PermissionMode mode)
{
  return get_underlying_value<PermissionMode>(mode);
}


} // namespace MessageQueue
} // namespace IPC

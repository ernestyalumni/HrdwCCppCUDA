ADD_SUBDIRECTORY(IO)
ADD_SUBDIRECTORY(Numerics)
ADD_SUBDIRECTORY(Std/Containers)
ADD_SUBDIRECTORY(Templates)
ADD_SUBDIRECTORY(Templates/FunctionT)
ADD_SUBDIRECTORY(Utilities)

ADD_LIBRARY(Cpp
  CallByValueCallByReference.cpp
  ClassAsEnum.cpp
  Classes/ConstructorsToDestructors.cpp
  RuleOf5.cpp)
ADD_SUBDIRECTORY(AlgebraicDataTypes)
ADD_SUBDIRECTORY(Functors)
# ADD_SUBDIRECTORY(Monoidal)
ADD_SUBDIRECTORY(Monads)

ADD_LIBRARY(Categories
	Monads/StateMonadExamples.cpp)

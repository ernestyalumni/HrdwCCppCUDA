"""
@file lec1.py
@url https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/lecture-slides-code/lec1.py
"""

class ComputeCircleArea:

	pi = 3.14159
	radius = 2.2
	# area of circle equation <- this is a comment
	area = pi*(radius**2)

class RecomputeCircleArea:
	# change values of radius <- another comment
	# use comments to help others understand what you are doing in code

	radius = ComputeCircleArea.radius + 1

	# area doesn't change
	area0 = ComputeCircleArea.area

	area = ComputeCircleArea.pi * (radius**2)



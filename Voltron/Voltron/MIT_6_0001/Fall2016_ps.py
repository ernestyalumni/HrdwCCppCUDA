"""
@file Fall2016_ps.py
@url https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/lecture-slides-code/lec1.py
"""

"""
Problem Set 0

@brief A Very Simple Program: Raising a number to a power and taking a
logarithm

@details 
"""
#from math import log

import numpy as np

def very_simple_program():
	"""
	@fn very_simple_program

	"""
	x = int(input("Enter number x:"))
	y = int(input("Enter number y:"))
	power_result = x**y
	print("X**y = " + str(power_result))
	log_result = np.log2(x)
	print("log(x) = " + str(log_result))

	return (power_result, log_result)
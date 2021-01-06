"""
@file test_MIT_OCW_6_0001_lec1.py

@url https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/lecture-slides-code/lec1.py

@details

EXAMPLE USAGE:

pytest ./test_MIT_OCW_6_0001_lec1.py
or
pytest test_MIT_OCW_6_0001_lec1.py

"""
from Voltron.MIT_6_0001 import lec1
from Voltron.MIT_6_0001.lec1 import (ComputeCircleArea, RecomputeCircleArea)

import pytest


@pytest.fixture
def circle_area_values():

	class ComputeCircleAreaValues:
		pi = ComputeCircleArea.pi
		radius = ComputeCircleArea.radius

	return ComputeCircleAreaValues()


def test_compute_circle_area(circle_area_values):
	pi = circle_area_values.pi
	radius = circle_area_values.radius
	# area of circle equation <= this is a comment
	area = pi*(radius**2)
	assert area == pytest.approx(15.20529560)


def test_recompute_circle_area(circle_area_values):
	pi = circle_area_values.pi
	radius = circle_area_values.radius
	# area of circle equation <= this is a comment
	area = pi*(radius**2)

	# @url https://stackoverflow.com/questions/39896716/can-i-perform-multiple-assertions-in-pytest
	errors = []
	if not (area == pytest.approx(15.20529560)):
		errors.append("area calc 1")

	#pytest.assume(area == pytest.approx(15.20529560))

	# change values of radius <- another comment
	# use comments to help others understand what you are doing in code
	radius = radius + 1
	# area doesn't change
	if not (area == pytest.approx(15.20529560)):
		errors.append("area did change")

	area = pi*(radius**2)

	assert not errors and area == pytest.approx(32.16988160)

#############################
#### AUTOCOMPLETE #######
#############################

def test_autocomplete():
	# define a variable
	a_very_long_variable_name_dont_name_them_this_long_pls = 0

	# below, start typing a_ve then hit tab... cool, right!
	# use autocomplete to change the value of that variable to 1

	a_very_long_variable_name_dont_name_them_this_long_pls = 1

	# use autocomplete to write a line that prints the value of that long
	# variable
	assert a_very_long_variable_name_dont_name_them_this_long_pls == 1

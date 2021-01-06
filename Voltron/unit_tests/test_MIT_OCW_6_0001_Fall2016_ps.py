"""
@file test_MIT_OCW_6_0001_Fall2016_ps.py

@url 

@details

EXAMPLE USAGE:

pytest ./test_MIT_OCW_6_0001_Fall2016_ps.py
or
pytest test_MIT_OCW_6_0001_Fall2016_ps.py

"""
from Voltron.MIT_6_0001 import Fall2016_ps
from Voltron.MIT_6_0001.Fall2016_ps import (very_simple_program)

from unittest import mock

import pytest

# Don't include the prefix; otherwise get error message: "Fixtures are not
# meant to be called directly."
#@pytest.fixture
def test_values():

	class TestValues:
		x = 2
		y = 3

	return TestValues()


def mock_input(input_prompt):
	"""
	@url https://stackoverflow.com/questions/35851323/how-to-test-a-function-with-input-call
	@url https://stackoverflow.com/questions/32682428/python-using-mock-for-a-multiple-user-inputs 
	"""
	if "x" in input_prompt.lower():
		return test_values().x
	if "y" in input_prompt.lower():
		return test_values().y


@mock.patch('builtins.input', mock_input)
def test_very_simple_program():
	"""
	@details

	Python 3 update:
	__builtin__ module is renamed to builtins in Python3.

	cf. https://stackoverflow.com/questions/18161330/using-unittest-mock-to-patch-input-in-python-3
	https://docs.python.org/3/library/builtins.html
	"""
	#with mock.patch.object(__builtins__, 'input', mock_input):
	assert very_simple_program() == (8, 1.0)
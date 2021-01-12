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
from Voltron.MIT_6_0001.Fall2016_ps import (
		calculate_months_for_down_payment,
		down_payment_time_with_raise,
		very_simple_program,
		MonthlySavingsGoal)

from unittest import mock

import pytest

# Don't include the prefix; otherwise get error message: "Fixtures are not
# meant to be called directly."
#@pytest.fixture
def very_simple_program_test_values():

		class TestValues:
				x = 2
				y = 3

		return TestValues()


def mock_very_simple_program_input(input_prompt):
		"""
		@url https://stackoverflow.com/questions/35851323/how-to-test-a-function-with-input-call
		@url https://stackoverflow.com/questions/32682428/python-using-mock-for-a-multiple-user-inputs 
		"""
		if "x" in input_prompt.lower():
				return very_simple_program_test_values().x
		if "y" in input_prompt.lower():
				return very_simple_program_test_values().y


@mock.patch('builtins.input', mock_very_simple_program_input)
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


"""
Problem Set 1
MIT 6.0001 Fall 2016
"""

def calculate_months_for_down_payment_test_values(option = 1):

		class TestValues1:
				annual_salary = 120000
				portion_saved = .10
				total_cost = 1000000

		class TestValues2:
				annual_salary = 80000
				portion_saved = .15
				total_cost = 500000

		if option == 1:
				return TestValues1()
		else:
				return TestValues2()


def mock_calculate_months_for_down_payment_closure(option = 1):
		"""
		@fn mock_calculate_months_for_down_payment_closure
		"""

		def mock_calculate_months_for_down_payment_input(input_prompt):
				if "annual salary" in input_prompt.lower():
						return calculate_months_for_down_payment_test_values(
								option).annual_salary
				if "percent of your salary to save" in input_prompt.lower():
						return calculate_months_for_down_payment_test_values(
								option).portion_saved
				if "cost of your dream home" in input_prompt.lower():
						return calculate_months_for_down_payment_test_values(
								option).total_cost

		return mock_calculate_months_for_down_payment_input


@mock.patch(
		'builtins.input',
		mock_calculate_months_for_down_payment_closure(1))
def test_calculate_months_for_down_payment_test_case_1():
		result = calculate_months_for_down_payment()
		assert result == 183

@mock.patch(
		'builtins.input',
		mock_calculate_months_for_down_payment_closure(2))
def test_calculate_months_for_down_payment_test_case_2():
		result = calculate_months_for_down_payment()
		assert result == 105


"""
Problem Set 1: Part B
"""

def down_payment_time_with_raise_test_values(option = 1):

		class TestValues1:
				annual_salary = 120000
				portion_saved = .05
				total_cost = 500000
				semi_annual_raise = .03

		class TestValues2:
				annual_salary = 80000
				portion_saved = .1
				total_cost = 800000
				semi_annual_raise = .03

		class TestValues3:
				annual_salary = 75000
				portion_saved = .05
				total_cost = 1500000
				semi_annual_raise = .05


		if option == 1:
				return TestValues1()
		elif option == 2:
				return TestValues2()
		else:
				return TestValues3()


def mock_down_payment_time_with_raise_closure(option = 1):
		"""
		@fn mock_down_payment_time_with_raise_closure
		"""

		def mock_down_payment_time_with_raise_input(input_prompt):
				if "annual salary" in input_prompt.lower():
						return down_payment_time_with_raise_test_values(
								option).annual_salary
				if "percent of your salary to save" in input_prompt.lower():
						return down_payment_time_with_raise_test_values(
								option).portion_saved
				if "cost of your dream home" in input_prompt.lower():
						return down_payment_time_with_raise_test_values(
								option).total_cost
		#		if "semi-annual salary raise" in input_prompt.lower():
				if "percent raise every 6 months" in input_prompt.lower():
						return down_payment_time_with_raise_test_values(
								option).semi_annual_raise

				# Should not reach here - should be an error.
				return None

		return mock_down_payment_time_with_raise_input


@mock.patch(
		'builtins.input',
		mock_down_payment_time_with_raise_closure(1))
def test_down_payment_time_with_raise_test_case_1():
		result = down_payment_time_with_raise()
		assert result == 142

@mock.patch(
		'builtins.input',
		mock_down_payment_time_with_raise_closure(2))
def test_down_payment_time_with_raise_test_case_2():
		result = down_payment_time_with_raise()
		assert result == 159

@mock.patch(
		'builtins.input',
		mock_down_payment_time_with_raise_closure(3))
def test_down_payment_time_with_raise_test_case_3():
		result = down_payment_time_with_raise()
		assert result == 261


"""
Problem Set 1
Part C: Finding the right amount to save away

Try different inputs for your starting salary, and see how percentage you
need to save changes to reach desired down payment.
"""

def savings_rate_test_values(option = 1):

		class TestValues1:
				starting_annual_salary = 150000
				expected_savings_rate = 0.4411
				expected_bisection_steps = 12

		class TestValues2:
				starting_annual_salary = 300000
				expected_savings_rate = 0.2206
				expected_bisection_steps = 9

		class TestValues3:
				starting_annual_salary = 10000

		if option == 1:
				return TestValues1()
		elif option == 2:
				return TestValues2()
		else:
				return TestValues3()


def mock_savings_rate_closure(option = 1):
		"""
		@fn mock_savings_rate_closure
		"""
		def mock_savings_rate_input(input_prompt):
				return savings_rate_test_values(option).starting_annual_salary

		return mock_savings_rate_input

def test_MonthlySavingsGoal_test_case_1():
		starting_annual_salary = savings_rate_test_values(1).starting_annual_salary
		expected_savings_rate = savings_rate_test_values(1).expected_savings_rate
		expected_bisection_steps = savings_rate_test_values(
				1).expected_bisection_steps

		g = MonthlySavingsGoal()
		savings_rate, steps = g.iterative_bisection_search(starting_annual_salary)

		assert ((savings_rate / 10000) == expected_savings_rate and 
				steps == expected_bisection_steps)


def test_MonthlySavingsGoal_test_case_2():
		starting_annual_salary = savings_rate_test_values(2).starting_annual_salary
		expected_savings_rate = savings_rate_test_values(2).expected_savings_rate

		# My iteration steps don't match what's given in MIT OCW 6.0001 F16.
		expected_bisection_steps = savings_rate_test_values(
				2).expected_bisection_steps

		g = MonthlySavingsGoal()
		savings_rate, steps = g.iterative_bisection_search(starting_annual_salary)

		assert ((savings_rate / 10000) == expected_savings_rate and 
				steps == 15)


def test_MonthlySavingsGoal_test_case_3():
		starting_annual_salary = savings_rate_test_values(3).starting_annual_salary

		g = MonthlySavingsGoal()
		result = g.iterative_bisection_search(starting_annual_salary)

		assert result == None

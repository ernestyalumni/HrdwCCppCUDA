"""
@file Fall2016_ps.py
@url https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-0001-introduction-to-computer-science-and-programming-in-python-fall-2016/lecture-slides-code/lec1.py
"""

import math # ps3
import random # ps3
import string # ps3

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

"""
Problem Set 1

@details Introduce you to using control flow in Python and formulating a
computational solution to a problem.
"""

class HouseSavingsAndInvestmentValues:
	# Portion of cost needed for down payment.
	# For simplicity, assume it's 0.25 (25%)
	portion_down_payment = 0.25

	# Annual return due to investment of your current savings.
	# Assume your investments return r = 0.04 (4%)
	r = 0.04


class HouseSavings:

	@staticmethod
	def monthly_return(current_savings, r):
		"""
		@brief Returns the additional funds to put into savings.
		"""
		return current_savings * r / 12

	@staticmethod
	def monthly_salary(annual_salary):
		"""
		@fn monthly_salary

		@details At end of each month, your savings will be increased by the
		return on your investment, plus percentage of your monthly salary
		(annual salary / 12)
		"""
		return annual_salary / 12

	def __init__(self):
		# Amount you've saved thus far. Start with 0.
		self.current_savings = 0

	def calculate_months_for_down_payment(self,
			annual_salary,
			portion_saved,
			# Cost of home.
			total_cost):
		portion_down = HouseSavingsAndInvestmentValues.portion_down_payment

		down_payment_needed = total_cost * portion_down

		months = 0
		while (down_payment_needed > self.current_savings):
			monthly_savings = self.calculate_monthly_savings(
				annual_salary,
				portion_saved)

			self.current_savings += monthly_savings
			months += 1

		return months

	def calculate_monthly_savings(
			self,
			annual_salary,
			portion_saved,
			return_rate = HouseSavingsAndInvestmentValues.r):
		"""
		@details This is a 2 part calculation.
		"""

		# Calculate savings from salary.

		monthly_salary_amount = HouseSavings.monthly_salary(annual_salary)
		# At the end of each month, savings increased by a percentage of your
		# monthly salary, portion_saved.
		monthly_salary_saved = monthly_salary_amount * portion_saved

		# Calculate return on investing savings.

		#return_rate = HouseSavingsAndInvestmentValues.r
		return_on_investment = HouseSavings.monthly_return(
			self.current_savings,
			return_rate)

		# At the end of each moth, savings increased by return on investment,
		# plus percentage of monthly salary
		return return_on_investment + monthly_salary_saved


def get_calculate_months_for_down_payment_inputs():
	annual_salary = float(input("Enter your annual salary: "))
	portion_saved = float(input(
		"Enter the percent of your salary to save, as a decimal: "))
	total_cost = float(input("Enter the cost of your dream home: "))

	return (annual_salary, portion_saved, total_cost)	


def calculate_months_for_down_payment():

	annual_salary, portion_saved, total_cost = \
		get_calculate_months_for_down_payment_inputs()

	house_savings = HouseSavings()
	months_needed = house_savings.calculate_months_for_down_payment(
		annual_salary,
		portion_saved,
		total_cost)

	print("Number of months: ", months_needed)
	return months_needed


"""
Problem Set 1
Part B: Saving, with a raise
"""

class HouseSavingsWithRaise:

	def __init__(self):
		self.house_savings = HouseSavings()


	def calculate_months_for_down_payment(self,
			annual_salary,
			portion_saved,
			# Cost of home.
			total_cost,
			semi_annual_raise):
		portion_down = HouseSavingsAndInvestmentValues.portion_down_payment

		down_payment_needed = total_cost * portion_down

		months = 0
		current_annual_salary = annual_salary

		while (down_payment_needed > self.house_savings.current_savings):

			# After the 6th month and after 12th month, 18th month, and so on.
			if (months > 0 and (months % 6 == 0)):
				current_annual_salary += (current_annual_salary
					* semi_annual_raise)

			# Reuse the calculation method from HouseSavings.
			monthly_savings = self.house_savings.calculate_monthly_savings(
				current_annual_salary,
				portion_saved)

			self.house_savings.current_savings += monthly_savings
			months += 1

		return months


def get_down_payment_time_with_raise_inputs():
	annual_salary, portion_saved, total_cost = \
		get_calculate_months_for_down_payment_inputs()

	# Decimal percentage raise every 6 months.
	semi_annual_raise = float(input(
		"Enter the percent raise every 6 months, as a decimal: "))

	return (annual_salary, portion_saved, total_cost, semi_annual_raise)


def down_payment_time_with_raise():

	annual_salary, portion_saved, total_cost, semi_annual_raise = \
		get_down_payment_time_with_raise_inputs()


	house_savings = HouseSavingsWithRaise()
	months_needed = house_savings.calculate_months_for_down_payment(
		annual_salary,
		portion_saved,
		total_cost,
		semi_annual_raise)

	print("Number of months: ", months_needed)
	return months_needed


"""
Problem Set 1
Part C: Finding the right amount to save away
"""

class MonthlySavingsGoalValues:
	# Semi-annual raise in decimal
	semi_annual_raise = .07
	# investment annual return
	r = 0.04
	# Down payment percent portion of cost of house.
	portion_down_payment = 0.25
	# Cost of the house that you're saving for
	total_house_cost = 1000000

	# How much to save?
	initial_portion_saved_lower_bound = 0
	initial_portion_saved_upper_bound = 10000

	# In months
	time_goal = 36

	# Within $100 or less
	upper_bound_error = 100


class MonthlySavingsGoal:

	def __init__(self):
		self.house_savings = HouseSavings()
		self.compute_down_payment()

	def compute_down_payment(self):

		self.down_payment_goal = (MonthlySavingsGoalValues.total_house_cost *
			MonthlySavingsGoalValues.portion_down_payment)

	def reset_current_savings(self):
		self.house_savings.current_savings = 0

	def compute_current_savings(self, annual_salary, portion_saved):

		semi_annual_raise = MonthlySavingsGoalValues.semi_annual_raise

		months = 0
		current_annual_salary = annual_salary

		while (months < MonthlySavingsGoalValues.time_goal):

			# After the 6th month and after 12th month, 18th month, and so on.
			if (months > 0 and (months % 6 == 0)):
				current_annual_salary += (current_annual_salary
					* semi_annual_raise)

			# Reuse the calculation method from HouseSavings.
			monthly_savings = self.house_savings.calculate_monthly_savings(
				current_annual_salary,
				portion_saved,
				MonthlySavingsGoalValues.r)

			self.house_savings.current_savings += monthly_savings
			months += 1

		current_savings = self.house_savings.current_savings
		self.reset_current_savings()

		return current_savings


	def initial_lower_bound_check(self, annual_salary):
		"""
		Returns false if down payment goal can be met without saving any salary
		"""
		portion_saved = \
			MonthlySavingsGoalValues.initial_portion_saved_lower_bound / 10000

		current_savings = self.compute_current_savings(
			annual_salary,
			portion_saved)

		return current_savings < self.down_payment_goal


	def initial_upper_bound_check(self, annual_salary):
		"""
		Returns false if down payment goal cannot be met at all.
		"""
		portion_saved = \
			MonthlySavingsGoalValues.initial_portion_saved_upper_bound / 10000

		current_savings = self.compute_current_savings(
			annual_salary,
			portion_saved)

		return current_savings >= self.down_payment_goal


	def iterative_bisection(self, lower_bound, upper_bound, annual_salary):
		if (lower_bound == upper_bound):
			return (lower_bound, upper_bound)

		average_in_decimal = (lower_bound + upper_bound) / 2.0 / 10000
		average_in_int = int((lower_bound + upper_bound) / 2)

		# Handle the case when they are only value 1 apart. Remember that int
		# casting truncates - it doesn't round.
		if (average_in_int == lower_bound):
			return (upper_bound, upper_bound)

		current_savings = self.compute_current_savings(
			annual_salary,
			average_in_decimal)

		if (current_savings >= self.down_payment_goal and
			(current_savings - self.down_payment_goal <
				MonthlySavingsGoalValues.upper_bound_error)):
			return (average_in_int, average_in_int)

		if (current_savings > self.down_payment_goal):
			return (lower_bound, average_in_int)
		# otherwise it must be that
		# current_savings < self.down_payment_goal
		else:
			return (average_in_int, upper_bound)


	def iterative_bisection_search(self, annual_salary):
		# Initialize lower and upper bounds.
		lower_bound = \
			MonthlySavingsGoalValues.initial_portion_saved_lower_bound		

		upper_bound = \
			MonthlySavingsGoalValues.initial_portion_saved_upper_bound

		if not self.initial_upper_bound_check(annual_salary):
			print("It is not possible to pay the down payment in three years")
			return None

		iterative_steps = 0

		while (lower_bound < upper_bound):
			lower_bound, upper_bound = self.iterative_bisection(
				lower_bound,
				upper_bound,
				annual_salary)

			iterative_steps += 1

			# Should terminate here.
			if (lower_bound == upper_bound):
				return (lower_bound, iterative_steps)

		return ((lower_bound, upper_bound), iterative_steps)



"""
Problem Set 3: The 6.00/6.0001 Word Game

@details Introduce you to using control flow in Python and formulating a
computational solution to a problem.
"""
class WordGameConstants:
		Vowels = 'aeiou'
		Consonants = 'bcdfghjklmnpqrstvwxyz'

		# TODO: Speed up this look up by calculating the index to look up.
		Scrabble_Letter_Values = {
    		'a': 1,
    		'b': 3,
    		'c': 3,
    		'd': 2,
    		'e': 1,
    		'f': 4,
    		'g': 2,
    		'h': 4,
    		'i': 1,
    		'j': 8,
    		'k': 5,
    		'l': 1,
    		'm': 3,
    		'n': 1,
    		'o': 1,
    		'p': 3,
    		'q': 10,
    		'r': 1,
    		's': 1,
    		't': 1,
    		'u': 1,
    		'v': 4,
    		'w': 4,
    		'x': 8,
    		'y': 4,
    		'z': 10}

		Wordlist_Filename = "words.txt"


# -----------------------------------
# Helper code
# (you don't need to understand this helper code)

class WordGameGivenHelperCode:

		@staticmethod
		def load_words():
				"""
				Returns a list of valid words. Words are strings of lowercase letters.
		
				Depending on the size of the word list, this function may take a while
				to finish.
				"""

				print("Loading word list from file...")
				# inFile: file
				inFile = open(WordGameConstants.Wordlist_Filename, 'r')
				# wordlist: list of strings
				wordlist = []
				for line in inFile:
						wordlist.append(line.strip().lower())
				print("  ", len(wordlist), "words loaded")
				return wordlist

		@staticmethod
		def get_frequency_dict(sequence):
				"""
				Returns a dictionary where the keys are elements of the sequence and
				the values are integer counts, for the number of times that an element
				is repeated in the sequence.

				sequence: string or list
				return: dictionary
				"""
				freq = {}

				return freq

# (end of helper code)
# -----------------------------------

# ps3.py
# Problem #1: Scoring a word
#

class PS3Problem1:

	@staticmethod
	def get_word_score(word, n):
		"""

		"""
		pass # TO DO... Remove this line when you implement this function
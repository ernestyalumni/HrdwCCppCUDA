"""
@file lecture_code.py
"""

"""
@file lec5_tuples_lists.py
"""

#########################
## EXAMPLE: returning a tuple
#########################

class ReturnTuple:

		@staticmethod
		def quotient_and_remainder(x, y):
				q = x // y
				r = x % y
				return (q, r)

		@staticmethod
		def compute_and_print(x, y):
				(quot, rem) = ReturnTuple.quotient_and_remainder(x, y)
				print(quot)
				print(rem)

		@staticmethod
		def example_run():
				compute_and_print(5,3)


#########################
## EXAMPLE: iterating over tuples
#########################

class IterateOverTuples:
		"""
		@brief Demonstrates we can iterate over a tuple.
		@url https://youtu.be/RvRKT-jXvko?t=641
		"""	

		test = ((1,"a"),(2,"b"),
				(1,"a"),(7,"b"))

		# apply get_data to any data you want!
		tswift = ((2014,"Katy"),
				(2014,"Harry"),
				(2012,"Jake"),
				(2010,"Taylor"),
				(2008,"Joe"))

		@staticmethod
		def get_data(aTuple):
				"""
				aTuple, tuple of tuples (int, string)
				Extracts, all integers from aTuple and sets
				them as elements in a new tuple.
				Extracts all unique strings from aTuple
				and sets them as elements in a new tuple.
				Returns a tuple of the minimum integer, the
				maximum integer, and the number of unique strings
				"""
				nums = () # empty tuple
				words = ()
				for t in aTuple:
						# concatenating with a singleton tuple
						nums = nums + (t[0],)
						# only add words haven't added before
						if t[1] not in words:
								words = words + (t[1],)
				min_n = min(nums)
				max_n = max(nums)
				unique_words = len(words)
				return (min_n, max_n, unique_words)

		@staticmethod 
		def get_data_test():
				(a, b, c) = IterateOverTuples.get_data(IterateOverTuples.test)
				print("a;",a,"b:",b,"c:",c)
				return (a, b, c)

		@staticmethod 
		def get_data_tswift():
				(min_year, max_year, num_people) = IterateOverTuples.get_data(
						IterateOverTuples.tswift)
				print("From", min_year, "to", max_year,
						"Taylor Swift wrote songs about", num_people, "people!")

				return (min_year, max_year, num_people)

#########################
## EXAMPLE: sum of elements in a list
#########################

class SumElements:
		"""
		@brief Demonstrates ways to iterate over a list.
		"""

		@staticmethod
		def sum_elem_method1(L):
				total = 0
				for i in range(len(L)):
						total += L[i]
				return total

		@staticmethod
		def sum_elem_method2(L):
				total = 0
				for i in L:
						total += i
				return total

		@staticmethod
		def test_method1(L = [1,2,3,4]):
				print(SumElements.sum_elem_method1(L))
				return SumElements.sum_elem_method1(L)

		@staticmethod
		def test_method2(L = [1,2,3,4]):
				print(SumElements.sum_elem_method2(L))
				return SumElements.sum_elem_method2(L)


#########################
## EXAMPLE: various list operations
## put print(L) at different locations to see how it gets mutated
#########################
class ListOperations:

		def __init__(self):
				self.L1 = [2,1,3]
				self.L2 = [4,5,6]
				self.L3 = self.L1 + self.L2
				self.L = [2,1,3,6,3,7,0]

				self.s = "I<3 cs"

		def extend_L1(self, x = [0,6]):
				self.L1.extend(x)

		def removing_elements(self, x = 2, y = 3):
				"""
				@details Also demonstrates mutability of a list.
				"""
				self.L.remove(x)
				self.L.remove(y)
				del(self.L[1])
				z = L.pop()
				print(z)
				return z

#########################
## EXAMPLE: aliasing
## EY: Aliasing is all about aliases. In the global frame or frame in scope,
## the 2 or more aliases point to the same object (for example list object)
#########################

class AliasExampleWithLists:
		def run_example():
				warm = ['red', 'yellow', 'orange']
				hot = warm
				hot.append('pink')
				return (hot, warm)

#########################
## EXAMPLE: cloning
#########################

class CloningIsDeepCopying:
		def __init__(self):
				self.cool = ['blue', 'green', 'grey']
				self.chill = self.cool[:]

				self.cl = ['b','gr','gy']
				self.ch = self.cl

		def mutate_chill_data_member(self):
				self.chill.append('black')

		def mutate_cl(self):
				self.cl.append('bl')


###############################
## EXAMPLE: mutating a list while iterating over it
###############################
class MutateWhileIterating:
		@staticmethod
		def remove_dups(L1, L2):
				# Python uses an internal counter to keep track of index it is in in
				# the loop.
				# Mutating changes list length but Python doesn't update the counter.
				for e in L1:
						if e in L2:
								L1.remove(e)

		@staticmethod
		def remove_dups_new(L1, L2):
				L1_copy = L1[:]
				for e in L1_copy:
						if e in L2:
								L1.remove(e)

		@staticmethod
		def remove_dups_v2(L1, L2):
				for e in L1[:]:
						if e in L2:
								L1.remove(e)


###############################
## EXERCISE: Test yourself by predicting what the output is and 
##           what gets mutated then check with the Python Tutor
###############################
def lecture_5_tuples_lists_mutation_example():
		cool = ['blue', 'green']
		warm = ['red', 'yellow', 'orange']
		print(cool)
		print(warm)

		colors1 = [cool]
		print(colors1)
		colors1.append(warm)
		print('colors1 = ', colors1)

		# TODO: complete this function if more practice is needed.


"""
@file lec6_recursion_dictionaries.py
"""

#####################################
# EXAMPLE: Towers of Hanoi
#####################################

class TowersOfHanoi:

	@staticmethod
	def print_move(frm, to):
		"""
		@param frm from
		"""
		print('move from ' + str(frm) + ' to ' + str(to))


	@staticmethod
	def towers(n, frm, to, spare):
		if n == 1:
			TowersOfHanoi.print_move(frm, to)
		else:
			TowersOfHanoi.towers(n - 1, frm, spare, to)
			TowersOfHanoi.towers(1, frm, to, spare)
			TowersOfHanoi.towers(n - 1, spare, to, frm)

	@staticmethod
	def print_example():
		result = TowersOfHanoi.towers(4, 'P1', 'P2', 'P3')
		print(result)
		return result
#print(towers(4, 'P1', 'P2', 'P3'))

#####################################
# EXAMPLE:  fibonacci
#####################################

def fib(x):
	"""
	assumes x an int >= 0
	returns Fibonacci of x
	"""
	if x == 0 or x == 1:
		return 1
	else:
		return fib(x-1) + fib(x-2)

#####################################
# EXAMPLE:  testing for palindromes
#####################################

def is_palindrome(s):

	def to_chars(s):
		s = s.lower()
		ans = ''
		for c in s:
			if c in 'abcdefghijklmnopqrstuvwxyz':
				ans = ans + c
		return ans

	def is_pal(s):
		if len(s) <= 1:
			return True
		else:
			return s[0] == s[-1] and is_pal(s[1:-1])

	return is_pal(to_chars(s))


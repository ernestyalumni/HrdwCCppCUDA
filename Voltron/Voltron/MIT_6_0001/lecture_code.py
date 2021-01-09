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
		return (min_n, max_n, unique_words)

	@staticmethod 
	def get_data_test():
		(a, b, c) = IterateOverTuples.get_data(IterateOverTuples.test)
		print("a;",a,"b:",b,"c:",c)
		return (a, b, c)

	@staticmethod 
	def get_data_tswift():
		return


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


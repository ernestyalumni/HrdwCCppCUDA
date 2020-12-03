"""
@file test_with_pytest.py

@ref https://realpython.com/pytest-python-testing/

@details

EXAMPLE USAGE:

pytest ./test_with_pytest.py
or
pytest test_with_pytest.py
"""

import pytest

# If you find yourself writing several tests that all make use of same
# underlying test data, then fxiture can pull repeated data into single
# function decorated with @pytest.fixture to indicate function is a pytest
# fixture:

@pytest.fixture
def example_people_data():
	return [
		{
			"given_name": "Alfonsa",
			"family_name": "Ruiz",
			"title": "Senior Software Engineer",
		},
		{
			"given_name": "Sayid",
			"family_name": "Khan",
			"title": "Project Manager",
		},
	]

# You can us fixture by adding it as argument to tests. Its value will be the
# return value of fixture function:

# def test_format_data_for_excel(example_people_data):
# 	assert format_data_for_excel(example_people_data) == """given,family,title
# Alfonsa,Ruiz,Senior Software Engineer
# Sayid,Khan,Project Manager
# """

def test_always_passes():
    assert True

# Uncomment out below to observe report of failure.
# unit_tests/test_with_pytest.py:12: AssertionError
#def test_always_fails():
#    assert False

def test_uppercase():
	assert "loud noises".upper() == "LOUD NOISES"

def test_reversed():
	assert list(reversed([1, 2, 3, 4])) == [4, 3, 2, 1]

def test_some_primes():
	assert 37 in {
		num
		for num in range(1, 50)
		if num != 1 and not any([num % div == 0 for div in range(2, num)])
	}


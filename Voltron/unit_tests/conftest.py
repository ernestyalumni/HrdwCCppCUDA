# @file conftest.py
#
# @details Guard access to resources with fixtures.

import pytest
import requests

# Suppose you want to ensure test suite doesn't make any real network calls,
# even if test accidentally executes real network call code.
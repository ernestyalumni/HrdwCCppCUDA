#!/bin/bash

# Sanity Check
pytest ./test_with_pytest.py

pytest Algorithms/test_level_easy.py

pytest DataStructures/test_ordered_symbol_table.py
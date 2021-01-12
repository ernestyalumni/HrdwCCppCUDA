

from Voltron.BuiltIn.super_inheritance import RightPyramid

import pytest

def test_RightPyramid_constructs_rectangle_data_members():
    r = RightPyramid(2, 4)
    assert(r.length == 2 and r.width == 2)

def test_RightPyramid_constructs_triangle_data_members():
    r = RightPyramid(2, 4)
    assert(r.height == 4)

def test_RightPyramid_area_computes_area():
    r = RightPyramid(2, 4)
    assert(r.area() == 20)

def test_RightPyramid_area_2_computes_area():
    r = RightPyramid(2, 4)
    assert(r.area_2() == 20)

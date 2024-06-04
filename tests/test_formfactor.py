"""Test the universal form factor module."""

import pytest
import sparapy.geometry as geo
import sparapy.form_factor as form_factor
import sparapy.ff_helpers.exact_solutions as exact_solutions

@pytest.mark.parametrize('width', [
    1.,2.,3.
    ])
@pytest.mark.parametrize('height', [
    1.,2.,3.
    ])
@pytest.mark.parametrize('distance', [
    1.,2.,3.
    ])
def test_parallel_facing_patches(width, height, distance):
    """Test universal form factor calculation for facing parallel patches of equal dimensions"""
    exact = exact_solutions.parallel_patches(width, height, distance)

    patch_1 = geo.Polygon(points=[[0,0,0],[width, 0, 0],[width, 0, height],[0,0,height]], normal=[0,1,0], up_vector=[1,0,0])

    patch_2 = geo.Polygon(points=[[0,distance,0],[width, distance, 0],[width, distance, height],[0,distance,height]], normal=[0,-1,0], up_vector=[1,0,0])

    computed = form_factor.calculate_form_factor(emitting_patch=patch_1, receiving_patch=patch_2)

    assert 100 * abs(computed-exact) < 1.

@pytest.mark.parametrize('width', [
    1.,2.,3.
    ])
@pytest.mark.parametrize('height', [
    1.,2.,3.
    ])
@pytest.mark.parametrize('length', [
    1.,2.,3.
    ])
def test_perpendicular_coincidentline_patches(width, height, length):
    """Test universal form factor calculation for perpendicular patches sharing a side"""

    exact = exact_solutions.perpendicular_patch_coincidentline(width,height,length)

    patch_1 = geo.Polygon(points=[[0,0,0],[width, 0, 0],[width, length, 0],[0,length,0]], normal=[0,0,1], up_vector=[1,0,0])

    patch_2 = geo.Polygon(points=[[0,0,0],[0, length, 0],[0, length, height],[0,0,height]], normal=[1,0,0], up_vector=[1,0,0])

    computed = form_factor.calculate_form_factor(emitting_patch=patch_1, receiving_patch=patch_2)

    assert 100 * abs(computed-exact) < 1.


@pytest.mark.parametrize('width1', [
    1.,2.,3.
    ])
@pytest.mark.parametrize('width2', [
    1.,2.,3.
    ])
@pytest.mark.parametrize('length1', [
    1.,2.,3.
    ])
@pytest.mark.parametrize('length2', [
    1.,2.,3.
    ])
def test_perpendicular_coincidentpoint_patches(width1, width2, length1, length2):
    """Test universal form factor calculation for perpendicular patches sharing a vertex"""

    exact = exact_solutions.perpendicular_patch_coincidentpoint(width1,length2,width2,length1)

    patch_1 = geo.Polygon(points=[[0,0,0],[0, length1, 0],[0, length1, width1],[0,0,width1]], normal=[1,0,0], up_vector=[0,1,0])

    patch_2 = geo.Polygon(points=[[0,length1,0],[width2, length1, 0],[width2, length1+length2, 0],[0,length1+length2,0]], normal=[0,0,1], up_vector=[1,0,0])

    computed = form_factor.calculate_form_factor(emitting_patch=patch_1, receiving_patch=patch_2)

    assert 100 * abs(computed-exact) < 1.
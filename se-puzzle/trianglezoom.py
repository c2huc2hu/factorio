from collections import namedtuple
import logging

from numpy import array, dot, cos, sin, pi
import numpy as np

logging.basicConfig(level=logging.INFO)

root_3 = np.sqrt(3)

Point = namedtuple('Point', ['x', 'y'])

# The triangle should be transformed such that the vertices are: (0,0), (1,0), (1/2, sqrt(3)/2)
# The point that the ray passes though the D60 is v

# We can find the subtriangle that the vector passes through by computing the row index counting
# from each vertex.


### Methods for finding where a point intersects the triangle ###

def find_row(v: array) -> int:
    """Find the row index of v counting from the top of the triangle"""
    # Find the component of v from the top of the triangle, whose height is root_3/2

    # scale height to 7 and count from the top
    return int(8 * (1 - v[1] / (root_3 / 2)))

def rotation_matrix_2d(theta: float) -> array:
    return array([[cos(theta), -sin(theta)], [sin(theta), cos(theta)]])

def find_subtriangle(v: array) -> tuple:
    # Each subtriangle is uniquely represented by the combination of its rows from each vertex.
    # The tuple is (rows from bottom left, rows from bottom right, rows from top)
    # This diagram makes more sense if you draw it on real paper, but:
    #     ...
    #         / 1,6,5 \ ...
    # / 0,6,6 \ 1,6,6 / 1,5,6 \ ...

    # Instead of redoing the math for the distance from the other corners, we can simply
    # rotate v by 120 degrees and use find_row.
    TRIANGLE_CENTER = array((1/2, root_3 / 4))
    v_rotated_cw = rotation_matrix_2d(-2 * pi / 3) @ v + array((1/2, root_3/2))
    v_rotated_ccw = rotation_matrix_2d(2 * pi / 3) @ v + array((1,0))

    row_1 = find_row(v_rotated_cw)
    row_2 = find_row(v_rotated_ccw)
    row_3 = find_row(v)
    subtriangle = (row_1, row_2, row_3)

    logging.info(f"v {v} is in subtriangle: {subtriangle}")

    return subtriangle


### Methods for zooming into the smaller triangle ###

def transform_v(v: array, subtriangle: tuple) -> array:
    # Transform from coordinates relative to the unit triangle to those relative to the smaller one

    is_inverted = bool(sum(subtriangle) % 2)

    if is_inverted:
        # Assumption: If the subtriangle is inverted, we flip (as opposed to rotate) the arrangement of subtriangles
        # I think I read something in the discord saying this is the case.

        # We can mirror v across the top edge of the inverted triangle,
        # and use the math from for the non-inverted triangle.
        top_edge = root_3/2 * 1/8 * (8 - subtriangle[2]) # NOTE: Every time you convert from row number to a point, you must multipy y by root_3/2
        v = array([v[0], 2*top_edge - v[1]])

        subtriangle = (subtriangle[0], subtriangle[1], subtriangle[2] - 1)

    bottom_left_corner_of_subtriangle = 1/8 * array([
        (2 * subtriangle[0] + subtriangle[2] - 7) / 2, # not intuitive, I had to make a table for this
        root_3/2 * (7 - subtriangle[2])
    ])
    new_v = 8 * (v - bottom_left_corner_of_subtriangle)

    logging.info(f"Transformed {v} in subtriangle {subtriangle} to {new_v}")

    assert 0 <= new_v[0] <= 1
    assert 0 <= new_v[1] <= root_3/2

    return new_v

def find_sequence(v: array) -> list:
    result = []
    for i in range(7):
        logging.info(f"Finding subtriangle for v={v}")
        subtriangle = find_subtriangle(v)
        result.append(subtriangle)
        v = transform_v(v, subtriangle)
    return result

if __name__ == '__main__':
    # Check that indexing is correct
    assert find_subtriangle(array([0.000001, 0.000001])) == (0, 7, 7)
    assert find_subtriangle(array([0.125, 0.125])) == (1, 7, 6)

    # Should all be center triangles
    center = array([0.5 + 0.001, root_3 / 4 + 0.001])
    # assert find_sequence(center) == [(5,5,3)] * 7


    # Possible guess:
    print("GUESS")
    guess = array([0.59686167, 0.24177929])
    print(find_sequence(guess))

    print("TEST")
    print(find_subtriangle(np.array([0.00001,0.00001])))
    print(find_subtriangle(np.array([0.000000001,0.000000001])))

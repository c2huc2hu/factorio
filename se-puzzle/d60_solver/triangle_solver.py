import numpy as np

from util import *

class TriangleSolver(object):
    def __init__(self, filename):
        self._read_glyph_pyramid(filename)

    def _read_glyph_pyramid(self, filename):
        self.glyph_pyramid = np.zeros([8,16], dtype=int) # [Row Column] indexing

        # Build the grid
        with open(filename) as fin:
            for row, line in enumerate(fin.readlines()):
                line = line.lstrip().rstrip()
                for col, glyph in enumerate(line.split(',')):
                    if glyph != "":
                        glyph = int(glyph)
                        self.glyph_pyramid[row][col] = glyph

        # Flip the rows so that the widest 'base' row is indexed 0
        self.glyph_pyramid = np.flip(self.glyph_pyramid, axis=0)

        assert np.sum(self.glyph_pyramid) == (1+64)*64/2

    def _get_point_subtriangle(self, triangle, point, flip_not_rotate=False, debug=False):
        # Triangle points are enumerated:
        #     2
        #    / \
        #   0---1

        # Consider point '0' of the triangle to be the origin.
        # Vectors 0->1 and 0->2 form a spanning basis:
        #    (0,1)
        #     / \
        # (0,0)-(1,0)

        if debug:
            print("Original Triangle")
            print(triangle)
            
            print("Target Point")
            print(target_point)

        # Reference all points to the origin of the triangle
        triangle_origin = np.array(triangle[0])
        adj_point = point - triangle_origin
        adj_triangle = triangle - triangle_origin

        if debug:
            print("Original Triangle (w.r.t Triangle Origin)")
            print(adj_triangle)

            print("Target Point (w.r.t. Triangle Origin)")
            print(adj_point)

        # Change-of-basis matrix:
        #   x_old = Ax_new 
        # where the columns of 'A' are the new basis vectors described in the old basis. 

        basis = np.transpose(np.array([
            (adj_triangle[1] - adj_triangle[0]),
            (adj_triangle[2] - adj_triangle[0]),
        ]))

        # After adjusting the basis, the triangle coordinates are now:
        #    (0,1)
        #     / \
        # (0,0)-(1,0)

        # Adjusting the conceptual frame of reference into this coordinage system, the triangle becomes a right-angle triangle:
        # (0,1)
        #   |  \
        # (0,0)-(1,0)

        # There are 64 equal triangles in the glyph pyramid, so the triangle is scaled up by a factor of 8 in each dimension
        # (ex. Scaling by a factor of 2):
        #  (0,2)
        #    |  \ 
        #  (0,1)-(1,1)
        #    |  \  |  \
        #  (0,0)-(1,0)-(2,0)

        # Dividing the basis vector by 8 before applying the change-of-basis accomplishes this:
        basis /= 8

        # Now transform the target point (referenced to the triangle's origin) into the new basis:
        # For the desired expression:
        #   x_new = A_invX_old we invert A
        basis_inv = np.linalg.inv(basis)

        # Bring the target point into the new basis
        new_basis_target_point = basis_inv.dot(adj_point)

        if debug:
            print("Target Point (New Basis)")
            print(new_basis_target_point)

        # Now, thinking in cartesian coordinates we need to:
        # 1. Locate the point in the square grid
        new_basis_target_cell = np.floor(new_basis_target_point)

        # 2. Identify if the point is in the upper or lower triangular half of the square
        #      If we re-define the point to be w.r.t. the lower-left corner of the target cell
        #      Then the diagonal separating the glyphs is between relative points (0,1) and (1,0).
        #      The equation of this line would be (x + y - 1 = 0).
        new_basis_target_in_upper_triangle = np.sum(new_basis_target_point - new_basis_target_cell) > 1

        # The [1] component describes which row of the pyramid we're in:
        glyph_row = int(new_basis_target_cell[1])

        # The [0] component describes which column of the pyramid we're in. 
        # Multiply the index by two since the upper and lower halves of the 
        # 'square' cell are in the same row of the pyramid array:
        glyph_column = int(2*new_basis_target_cell[0])

        # Finally, if the point is in the upper half-triangle, shift the column one to the left
        glyph_column += 1 if new_basis_target_in_upper_triangle else 0

        # Pick the glyph from the pyramid array
        if debug:
            print("Checking Glyph Pyramid at ({}, {}) = ".format(glyph_row, glyph_column), end="")

        glyph = self.glyph_pyramid[glyph_row][glyph_column]

        if debug:
            print(glyph)

        assert glyph > 0

        # Now identify the new triangle:
        if new_basis_target_in_upper_triangle:
            # If the new triangle is an upper-half triangle then it is in a different orientation w.r.t the original triangle.
            # The base of the triangle is always the horizontal line
            # The peak of the triangle is always given last
            if flip_not_rotate:
                # Flip across the base, points are now given in clockwise order
                new_triangle = [ [0,1], [1,1], [1,0] ]
            else:
                # Rotate around the triangle origin, points are still given in anti-clockwise order:
                new_triangle = [ [1,1], [0,1], [1,0] ]
        else:
            # If the new triangle is one of the three 'right-side-up' triangles, then its coordinates are simply:
            new_triangle = [ [0,0], [1,0], [0,1] ] 

        if debug:
            print("New Triangle:")
            print(new_triangle)

        # Shift the new triangle to the cartesian cell corresponding to the glyph:
        new_triangle += new_basis_target_cell

        if debug:
            print("New Triangle (Shifted):")
            print(new_triangle)

        # Transform the triangle back into the original basis:
        new_triangle = np.transpose(basis.dot(np.transpose(new_triangle)))

        if debug:
            print("New Triangle (Original Basis):")
            print(new_triangle)

        # Relocate the new triangle w.r.t. the original triangle's origin
        new_triangle += triangle_origin

        if debug:
            print("New Triangle (Final):")
            print(new_triangle)

        # Return the refined triangle and the corresponding glyph
        return new_triangle, glyph

            
    ################################################################################
    # Public Functions
    ################################################################################
    def get_point_from_sequence(self, triangle, seq):
        pass

    def get_sequence_from_point(self, triangle, target_point, steps):
        # Copy the triangle so as to not change the original
        new_triangle = np.array(triangle)

        sequence = []
        triangles = []

        for _ in range(steps):
            new_triangle, glyph = self._get_point_subtriangle(new_triangle, target_point)
            sequence.append(glyph)
            triangles.append(np.array(new_triangle))

        return sequence, triangles
        

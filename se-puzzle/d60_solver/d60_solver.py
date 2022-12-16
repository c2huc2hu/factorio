import numpy as np
from skspatial.objects import Plane

from util import *

class D60Solver(object):
    DEFAULT_REFERENCE_BASIS = [
        np.array([0, 1, 3*PHI]),
        np.array([1, 2+PHI, 2*PHI]),
        np.array([PHI, 2, 2*PHI+1])
    ]

    ORIGIN = np.array([0, 0, 0], dtype=np.float64)

    def __init__(self, reference_basis=DEFAULT_REFERENCE_BASIS, starmapping_file="starmapping.dat"):
        self.glyph_known_vectors = np.zeros([60, 3], dtype=np.float64)
        self.glyph_adj = np.zeros([60, 3], dtype=int) 
        self.point_adj = np.zeros([60, 3], dtype=int)
        self.glyph_names = ["" for _ in range(60)]

        self._create_truncated_icosahedron(reference_basis)
        self._find_adj_points()
        self._read_glyph_data(starmapping_file)
        self._associate_glyph_points()

    def _create_truncated_icosahedron(self, ref_vectors):
        """
        Obtain the points of a Truncated Icosahedron (dual Polyhedron of Pentakis Dodecahedron (D60))
        using a given initial set of 3 'reference vectors'. 

        Ref: https://en.wikipedia.org/wiki/Truncated_icosahedron#Cartesian_coordinates
        """
        tmp_points = []
        final_points = []

        for i_point in ref_vectors:
            tmp_points += even_permutation_3(i_point)

        for j_point in tmp_points:
            final_points += sign_permutation_3(j_point)

        self.points = np.unique(np.array(final_points), axis=0)

    def _verify_adj(self, connectivity):
        # Verify that adjacencies are reciprocal
        for node, adjacent in enumerate(connectivity):
            # 'C' adjacency should be reciprocal
            assert (node == connectivity[int(adjacent[2])][2]), "C Adjacency Error Node: {}".format(node)

            # 'A'/'B' adjacency should be cyclic
            assert (node == connectivity[int(adjacent[0])][1]), "A Adjacency Error Node: {}".format(node)
            assert (node == connectivity[int(adjacent[1])][0]), "B Adjacency Error Node: {}".format(node)

        # Verify that the sum of each column is such that each value appears exactly once
        assert np.all(np.sum(connectivity, axis=0) == ((len(connectivity)-1) * len(connectivity)/2))

    def _read_glyph_data(self, filename):
        """
        Read a .csv file with the following fields:
            A, B, C, Name, X, Y, Z

            [A, B, C] are the glyphs adjacent to the given glyph. 
                [A, B] have cyclic symmetry
                [C] has reciprocal symmetry.

            [Name] is the name of the planet/system where the monument for this glyph was discovered

            [X, Y, Z] is the spatial vector for this glyph, obtained in-game via the Starmapping Research
        """
        with open(filename) as fin:
            for i, line in enumerate(fin.readlines()):
                data = line.split(',')

                # Glyphs are 1-indexed in the file, 0-index them here:
                self.glyph_adj[i] = np.array(data[:3], dtype=int) - 1
                self.glyph_names[i] = data[3]

                try:
                    self.glyph_known_vectors[i] = np.array(data[4:], dtype=np.float64)
                except ValueError:
                    pass

            self._verify_adj(self.glyph_adj)

    def _find_adj_points(self):
        """
        Find the three nearest-neighbours of every point in the polyhedron
        """
        self.points = normalize(self.points)

        for i, point in enumerate(self.points):
            dot_products = []
            
            for other_point in self.points:
                dot_products.append(np.dot(point, other_point))

            # Largest dot products are the closest points
            # Dot product with itself will always be 1, so the next three are the adjacent points
            adjacent_points = np.flip(np.argsort(dot_products))[1:4]

            # One of these things is not like the other. Since D60 sides are isosceles, one neighbour is the 'Base' 
            # and the other two are the 'Sides'. For consistency, reorder the list so that the 'Base' is last.

            # The pair-wise dot-products between the neighbours determine which side is the 'base' adjacency
            pairwise_dot_products = [
                np.dot(self.points[adjacent_points[1]], self.points[adjacent_points[2]]), # Point 0 is not involved
                np.dot(self.points[adjacent_points[2]], self.points[adjacent_points[0]]), # Point 1 is not involved
                np.dot(self.points[adjacent_points[0]], self.points[adjacent_points[1]]), # Point 2 is not involved
            ]

            # The dot product between the two side vectors will be larger than the one between either side an the base. 
            # Use the magnitudes of the dot products to reorder the neighbour points.

            # To do this, argsort the pariwise list and then use the result as the indices into the adjacent_points list:
            adjacent_points = adjacent_points[np.argsort(pairwise_dot_products)]

            # Then, to keep the ordering of the sides consistent, swap the sides if the dot product of their cross product with
            # the original point is negative
            dot_of_cross = np.dot(np.cross(self.points[adjacent_points[0]], self.points[adjacent_points[1]]), point)

            if dot_of_cross < 0:
                adjacent_points = adjacent_points[np.array([1, 0, 2])]

            self.point_adj[i] = np.array(adjacent_points, dtype=int)

        self._verify_adj(self.point_adj)

    def _associate_glyph_points(self):
        """
        Rearranges the points on the polyhedron into the order specified by the glyphs.
        """
        assert len(self.glyph_adj) == len(self.point_adj), "{} != {}".format(len(self.glyph_adj), len(self.point_adj))

        n = len(self.point_adj)

        self.glyph_points = np.ones(n, dtype=int) * -1

        glyphs = set(range(n))

        # Coordinates of some glyphs are known
        known_glyphs = set(list(np.where(np.any(self.glyph_known_vectors != 0, axis=1))[0]))
        hidden_glyphs = set()

        # Remove some glyphs from the known set to check against later
        for _ in range(len(known_glyphs)//2):
            hidden_glyphs.add(known_glyphs.pop())

        # Find the closest points each known glyph
        while len(known_glyphs) > 0:
            current_glyph = known_glyphs.pop()

            glyph_coords = self.glyph_known_vectors[current_glyph]

            distances = np.ones(n) * np.inf

            for i, point in enumerate(self.points):
                distances[i] = np.linalg.norm(point - glyph_coords)

            closest_point = np.argmin(distances)

            # print("Star {} ({}) : {} is closest to point ({}) {}".format(self.glyph_names[current_glyph], current_glyph, self.glyph_known_vectors[current_glyph], closest_point, points[closest_point]))

            # Associate the nearest point to this glpyh
            self.glyph_points[current_glyph] = closest_point

        # Now fill in the rest of the associations by adjacency
        known_glyphs = set(list(np.where(np.any(self.glyph_known_vectors != 0, axis=1))[0])).difference(hidden_glyphs)

        current_glyph = None
        next_glyphs = set()

        while len(glyphs) > 0:
            # Start with a known glyph
            if len(next_glyphs) == 0:
                current_glyph = known_glyphs.pop()
            else:
                current_glyph = next_glyphs.pop()

            current_point = self.glyph_points[current_glyph]
            assert current_point != -1

            # Associate all neighbour glyphs with the neighbour points
            for adj_glyph, adj_point in zip(self.glyph_adj[current_glyph], self.point_adj[current_point]):
                if self.glyph_points[adj_glyph] == -1:
                    self.glyph_points[adj_glyph] = adj_point
                else:
                    assert self.glyph_points[adj_glyph] == adj_point

                if adj_glyph in glyphs:
                    next_glyphs.add(adj_glyph)

            glyphs.remove(current_glyph)

        # Check against the hidden glyphs
        while len(hidden_glyphs) > 0:
            check_glyph = hidden_glyphs.pop()

            # print("Compare Glyph {} : {} Estimated : {} Actual : {}".format(check_glyph, self.glyph_names[check_glyph], self.points[self.glyph_points[check_glyph]], self.glyph_known_vectors[check_glyph]))
            assert np.dot(self.points[self.glyph_points[check_glyph]], self.glyph_known_vectors[check_glyph]) > 0.999

    ################################################################################
    # Public Functions
    ################################################################################
    @property
    def vectors(self):
        """
        Return the list of spatial points in glyph order
        """
        return self.points[self.glyph_points]
        
    def get_nearest_point_vector(self, target_vector):
        dot_products = np.zeros(len(self.points))
        for i, point in enumerate(self.points):
            dot_products[i] = np.dot(target_vector, point)

        point_idx = np.argmax(dot_products)

        return point_idx, self.points[point_idx]

    def get_glyph_face_vector(self, glyph):
        return self.vectors[glyph]

    def get_glyph_face_triangle(self, glyph):
        starting_point = self.points[self.glyph_points[glyph]]
        adjacent_points = self.points[self.point_adj[self.glyph_points[glyph]]]

        starting_plane = Plane(point=starting_point, normal=starting_point)

        triangle = np.zeros([3,3], dtype=np.float64)

        for i in range(3):
            adj_face_0 = Plane(point=adjacent_points[(i+1)%3], normal=adjacent_points[(i+1)%3])
            adj_face_1 = Plane(point=adjacent_points[(i+2)%3], normal=adjacent_points[(i+2)%3])
            triangle[i] = starting_plane.intersect_line(adj_face_0.intersect_plane(adj_face_1))

        print(triangle)
        return triangle

    def plot(self, ax):
        # Plot Points
        plot_points_3D(ax, self.points, "", 8.0)

        # Plot Edges
        for point, neighbours in enumerate(self.point_adj):
            for i, neighbour in enumerate(neighbours):
                draw_line_3D(ax, self.points[point], self.points[neighbour], 'r' if i == 2 else 'b')

        # Plot Known Stars in Magenta
        plot_points_3D(ax, self.glyph_known_vectors[np.where(np.any(self.glyph_known_vectors != 0, axis=1))], '*m', 16)

        # Plot the inferred star points in cyan
        plot_points_3D(ax, self.vectors[np.where(np.all(self.glyph_known_vectors == 0, axis=1))], '*c', 16)

        # Label all star points
        for glyph in range(len(self.points)):
            ax.text(self.points[glyph][0], self.points[glyph][1], self.points[glyph][2], self.glyph_names[glyph])

    def write_starmapping_file(self, filename):
        with open(filename, 'w+') as fout:
            for glyph in range(len(self.points)):
                output_list = list(self.glyph_adj[glyph]) 
                output_list.append(self.glyph_names[glyph])
                output_list += list(self.vectors[glyph])
                fout.write(",".join([str(x) for x in output_list])+"\n")


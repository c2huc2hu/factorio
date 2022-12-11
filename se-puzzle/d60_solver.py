from stl import mesh
import collections
import numpy as np
from scipy.spatial.transform import Rotation
import matplotlib.pyplot as plt

ORIGIN = np.array([0, 0, 0], dtype=np.float64)

PHI = (1 + 5 ** 0.5) / 2

DEFAULT_REFERENCE_BASIS = [
    np.array([0, 1, 3*PHI]),
    np.array([1, 2+PHI, 2*PHI]),
    np.array([PHI, 2, 2*PHI+1])
]

OBSERVED_REFERENCE_BASIS = [
    np.array([0,            0.1898387955, 0.9818152737]),
    np.array([0.2084778207, 0.7356418278, 0.6444904486]),
    np.array([0.3373248251, 0.3983170027, 0.8529686558]),
]

DESTINATION_VECTOR = np.array([0.3994985285, -0.7633068197, 0.507704269])

GLYPH_NAMES = ["" for _ in range(60)]
STARMAPPING = np.zeros([60,3])

def unit(vec):
    return vec / np.linalg.norm(vec)

# Pentakis Dodecahedron (D60) is the overall shape of the sky
# Dual of the Pentakis Dodecahedron is the Truncated Icosahedron (60 vertices)
def sign_permutation_3(point):
    points = []
    for i in [-1, 1]:
        for j in [-1, 1]:
            for k in [-1, 1]:
                points.append(np.multiply(point, [i, j, k]))

    return points

def even_permutation_3(point):
    # Yeah, you could probably do this programmatically, but it's eluding me right now
    points = [
        np.array([point[0], point[1], point[2]]),
        np.array([point[1], point[2], point[0]]),
        np.array([point[2], point[0], point[1]])
    ]

    return points

def truncated_icosahedron(ref_vectors=DEFAULT_REFERENCE_BASIS):
    
    tmp_points = []
    final_points = []

    for i_point in ref_vectors:
        tmp_points += even_permutation_3(i_point)

    for j_point in tmp_points:
        final_points += sign_permutation_3(j_point)

    return np.unique(np.array(final_points), axis=0)

def normalize(points):
    return np.array([unit(vec) for vec in list(points)])

def plot_points(ax, points, style="", markersize=1.0):
    ax.plot(points[:,0], points[:,1], points[:,2], style, linestyle="None", markersize=markersize)

def draw_line(ax, a, b, style=""):
    ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], style);

def verify_connectivity(connectivity):
    # Verify that adjacencies are reciprocal
    for node, adjacent in enumerate(connectivity):
        # 'C' adjacency should be reciprocal
        assert (node == connectivity[int(adjacent[2])][2]), "C Adjacency Error Node: {}".format(node)

        # 'A'/'B' adjacency should be cyclic
        assert (node == connectivity[int(adjacent[0])][1]), "A Adjacency Error Node: {}".format(node)
        assert (node == connectivity[int(adjacent[1])][0]), "B Adjacency Error Node: {}".format(node)

    # Verify that the sum of each column is such that each value appears exactly once
    assert np.all(np.sum(connectivity, axis=0) == ((len(connectivity)-1) * len(connectivity)/2))

def find_adjacent_points(points):
    adjacencies = []

    points = normalize(points)

    for i, point in enumerate(points):
        dot_products = []
        
        for other_point in points:
            dot_products.append(np.dot(point, other_point))

        # Largest dot products are the closest points
        # Dot product with itself will always be 1, so the next three are the adjacent points
        adjacent_points = np.flip(np.argsort(dot_products))[1:4]

        # One of these things is not like the other. Since D60 sides are isosceles, one neighbour is the 'Base' 
        # and the other two are the 'Sides'. For consistency, reorder the list so that the 'Base' is last.

        # The pair-wise dot-products between the neighbours determine which side is the 'base' adjacency
        pairwise_dot_products = [
            np.dot(points[adjacent_points[1]], points[adjacent_points[2]]), # Point 0 is not involved
            np.dot(points[adjacent_points[2]], points[adjacent_points[0]]), # Point 1 is not involved
            np.dot(points[adjacent_points[0]], points[adjacent_points[1]]), # Point 2 is not involved
        ]

        # The dot product between the two side vectors will be larger than the one between either side an the base. 
        # Use the magnitudes of the dot products to reorder the neighbour points.

        # To do this, argsort the pariwise list and then use the result as the indices into the adjacent_points list:
        adjacent_points = adjacent_points[np.argsort(pairwise_dot_products)]

        # Then, to keep the ordering of the sides consistent, swap the sides if the dot product of their cross product with
        # the original point is negative
        dot_of_cross = np.dot(np.cross(points[adjacent_points[0]], points[adjacent_points[1]]), point)

        if dot_of_cross < 0:
            adjacent_points = adjacent_points[np.array([1, 0, 2])]

        adjacencies.append(adjacent_points)

    print(np.array(adjacencies))

    verify_connectivity(adjacencies)

    return adjacencies

def read_glyph_data():
    # Connectivity of a 60-node graph
    connectivity = np.zeros([60,3], dtype=int)
    graph = np.zeros([60,60], dtype=int)

    # Read the .dat file
    with open("graph.dat") as fin:
        for i, line in enumerate(fin.readlines()):
            data = line.split(',')

            # Adjust to 0-indexed Glpyh IDs
            connectivity[i] = np.array(data[:3], dtype=int) - 1
            GLYPH_NAMES[i] = data[3].lstrip().rstrip()

            try:
                STARMAPPING[i] = np.array(data[4:], dtype=np.float64)
            except ValueError:
                pass

    verify_connectivity(connectivity)

    return connectivity

def associate_glyph_points(adjacent_glyphs, points, adjacent_points):
    # Arguments:
    #   adjacent_glyphs: a [60,3] array of glyph adjacencies. Note the lack of rotational symmetry.
    #   adjacent_points: a [60,3] array of point adjacencies. Note the lack of rotational symmetry.

    # Returns the list of point indices corresponding to the glyph order
    assert len(adjacent_glyphs) == len(adjacent_points), "{} != {}".format(len(adjacent_glyphs), len(adjacent_points))

    n = len(adjacent_points)

    glyph_points = np.ones(n, dtype=int) * -1

    glyphs = set(range(n))

    # Coordinates of some glyphs are known
    known_glyphs = set(list(np.where(np.any(STARMAPPING != 0, axis=1))[0]))
    hidden_glyphs = set()

    # Remove some glyphs from the known set to check against later
    for _ in range(len(known_glyphs)//2):
        hidden_glyphs.add(known_glyphs.pop())

    # Find the closest points each known glyph
    while len(known_glyphs) > 0:
        current_glyph = known_glyphs.pop()

        glyph_coords = STARMAPPING[current_glyph]

        distances = np.ones(n) * np.inf

        for i, point in enumerate(points):
            distances[i] = np.linalg.norm(point - glyph_coords)

        closest_point = np.argmin(distances)

        print("Star {} ({}) : {} is closest to point ({}) {}".format(GLYPH_NAMES[current_glyph], current_glyph, STARMAPPING[current_glyph], closest_point, points[closest_point]))

        # Associate the nearest point to this glpyh
        glyph_points[current_glyph] = closest_point

    # Now fill in the rest of the associations by adjacency
    known_glyphs = set(list(np.where(np.any(STARMAPPING != 0, axis=1))[0])).difference(hidden_glyphs)

    current_glyph = None
    next_glyphs = set()

    while len(glyphs) > 0:
        # Start with a known glyph
        if len(next_glyphs) == 0:
            current_glyph = known_glyphs.pop()
        else:
            current_glyph = next_glyphs.pop()

        current_point = glyph_points[current_glyph]
        assert current_point != -1

        # Associate all neighbour glyphs with the neighbour points
        for adj_glyph, adj_point in zip(adjacent_glyphs[current_glyph], adjacent_points[current_point]):
            if glyph_points[adj_glyph] == -1:
                glyph_points[adj_glyph] = adj_point
            else:
                assert glyph_points[adj_glyph] == adj_point

            if adj_glyph in glyphs:
                next_glyphs.add(adj_glyph)

        glyphs.remove(current_glyph)

    # Check against the hidden glyphs
    while len(hidden_glyphs) > 0:
        check_glyph = hidden_glyphs.pop()

        print("Compare Glyph {} : {} Estimated : {} Actual : {}".format(check_glyph, GLYPH_NAMES[check_glyph], points[glyph_points[check_glyph]], STARMAPPING[check_glyph]))
        assert np.dot(points[glyph_points[check_glyph]], STARMAPPING[check_glyph]) > 0.999

    return glyph_points


################################################################################
# Main
################################################################################
def main():
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    fig.subplots_adjust(top=1, bottom=0)
    ax.set_xlim3d(-1,1)
    ax.set_ylim3d(-1,1)
    ax.set_zlim3d(-1,1)

    # Draw the target course
    destination = unit(DESTINATION_VECTOR) * 10
    draw_line(ax, ORIGIN, destination, 'g')

    # Calculate the D60
    points = truncated_icosahedron(OBSERVED_REFERENCE_BASIS)

    plot_points(ax, points, "", 8.0)

    adjacent_points = find_adjacent_points(points)

    for point, neighbours in enumerate(adjacent_points):
        for i, neighbour in enumerate(neighbours):
            draw_line(ax, points[point], points[neighbour], 'r' if i == 2 else 'b')


    adjacent_glyphs = read_glyph_data()

    # Map the glyphs to points
    glyph_points = associate_glyph_points(adjacent_glyphs, points, adjacent_points)

    # Reorder the list of vectors based on the glyph/point association
    vectors = points[glyph_points]

    # Plot the known star points in magenta
    plot_points(ax, STARMAPPING[np.where(np.any(STARMAPPING != 0, axis=1))], '*m', 16)

    # Plot the inferred star points in cyan
    plot_points(ax, vectors[np.where(np.all(STARMAPPING == 0, axis=1))], '*c', 16)

    # Label all star points
    for glyph in range(60):
        ax.text(vectors[glyph][0], vectors[glyph][1], vectors[glyph][2], GLYPH_NAMES[glyph])

    # Find the closest vector to the target vector
    dot_products = np.zeros(60)

    for i, vector in enumerate(vectors):
        dot_products[i] = np.dot(vector, DESTINATION_VECTOR)

    initial_vector = np.argmax(dot_products)

    print("Initial Vector: Glyph {} : {} : {}".format(initial_vector, GLYPH_NAMES[initial_vector], vectors[initial_vector]))

    draw_line(ax, ORIGIN, unit(vectors[initial_vector])*10, 'g--')

    # Dump the data for the spreadsheet
    for glyph, point in enumerate(glyph_points):
        # 1-Indexed to match the spreadsheet
        print(glyph + 1, " : ", points[int(point)])

    plt.show()
    
if __name__ == "__main__":
    main()

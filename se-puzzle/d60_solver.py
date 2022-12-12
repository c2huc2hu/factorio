from stl import mesh
import collections
import numpy as np
from scipy.spatial.transform import Rotation
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

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

CORNER_VECTORS = [
    np.array([1.7533007652899e-07, -0.93417235727245, 0.35682209419822]), # Vector [44,61,61,61,61,61,61,61]
    np.array([0.57735021142964, -0.57735041155694, 0.57735018458227]), # Vector [44,62,62,62,62,62,62,62]
    np.array([0.52573107107952, -0.8506508337159, 1.4848270733249e-07]), # Vector [44,63,63,63,63,63,63,63]
]

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

def line_plane_intersection(line_vec, norm_vec, line_point, plane_point):
    d = np.dot((plane_point - line_point), norm_vec) / np.dot(line_vec, norm_vec)
    return line_point + line_vec * d

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

def plot_point_3D(ax, point, style="", markersize=1.0):
    ax.plot(point[0], point[1], point[2], style, markersize=markersize, linestyle="None")

def plot_points_3D(ax, points, style="", markersize=1.0):
    ax.plot(points[:,0], points[:,1], points[:,2], style, linestyle="None", markersize=markersize)

def draw_line_3D(ax, a, b, style=""):
    ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]], style);

def plot_point_2D(ax, point, style, markersize=1.0):
    ax.plot(point[0], point[1], style, markersize=markersize, linestyle="None")

def plot_points_2D(ax, points, style="", markersize=1.0):
    ax.plot(points[:,0], points[:,1], style, linestyle="None", markersize=markersize)

def draw_line_2D(ax, a, b, style=""):
    ax.plot([a[0], b[0]], [a[1], b[1]], style);

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

def refine_triangular_segment(triangle, target_point):
    # Triangle points are enumerated:
    #     2
    #    / \
    #   0---1

    # Consider point '0' of the triangle to be the origin.
    # Vectors 0->1 and 0->2 form a spanning basis:
    #    (0,1)
    #     / \
    # (0,0)-(1,0)

    # Reference all points to the origin of the triangle
    triangle_origin = np.array(triangle[0])
    adj_target_point = target_point - triangle_origin
    adj_triangle = triangle - triangle_origin

    # Change-of-basis matrix:
    #   x_old = Ax_new 
    # where the columns of 'A' are the new basis vectors described in the old basis. 

    basis = np.transpose(np.array([
        (adj_triangle[1] - adj_triangle[0]),
        (adj_triangle[2] - adj_triangle[0]),
    ]))

    # To refine our guess, subdivide the triangle into four segments (A,B,C,D)
    #         (0,2)
    #         / C \
    #     (0,1)---(1,1)
    #     / A \ D / B \
    # (0,0)---(1,0)---(2,0)
    # 
    # Dividing the basis vectors by 2 before the change-of-basis provides this transform
    basis /= 2

    # For the desired expression:
    #   x_new = A_invX_old we invert A
    basis_inv = np.linalg.inv(basis)

    # Bring the target point into the new basis
    new_basis_target_point = basis_inv.dot(adj_target_point)

    # The refined triangle uses the points of the triangular segment, 
    # in the same counter-clockwise order starting across the base. 
    refined_triangles = [
        np.array([[0,0],[1,0],[0,1]]), # A
        np.array([[1,0],[2,0],[1,1]]), # B
        np.array([[0,1],[1,1],[0,2]]), # C
        np.array([[1,1],[0,1],[1,0]]), # D
    ]

    x, y = new_basis_target_point

    assert x > 0
    assert y > 0

    if x + y < 1:
        triangle_idx = 0 # 'A'
    elif x + y < 2:
        if x > 1:
            triangle_idx = 1 # 'B'
        elif y > 1:
            triangle_idx = 2 # 'C'
        else:
            triangle_idx = 3 # 'D'
    else:
        assert False

    # Transform the new triangular segment back to the original basis
    new_triangle = np.transpose(basis.dot(np.transpose(refined_triangles[triangle_idx])))

    # Restore the original origin
    new_triangle += triangle_origin

    # Return the refined triangle, and which segment it was of the original triangle
    return new_triangle, triangle_idx

def test_refine_triangle():
    fig = plt.figure()
    ax = fig.add_subplot()
    ax.set_xlim(0,1)
    ax.set_ylim(0,1)

    triangle = np.array([
        [0,0],
        [1,0],
        [0.5, np.sqrt(3)/2]
    ])

    # Random Points
    points = np.random.uniform(size=(100,2))

    colors = ['c', 'm', 'y', 'k']

    num_points = 0

    while num_points < 10000:
        point = np.random.uniform(size=2)

        try:
            idx = refine_triangular_segment(triangle, point)
        except AssertionError:
            continue
        
        color = colors[idx]

        ax.plot(point[0], point[1], '{}o'.format(color), linestyle='None')

        num_points += 1

    fig.show()


################################################################################
# Main
################################################################################
def main():
    # Plotting
    fig = plt.figure()
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    #fig.subplots_adjust(top=1, bottom=0)
    ax.set_xlim3d(-1,1)
    ax.set_ylim3d(-1,1)
    ax.set_zlim3d(-1,1)

    n = 60

    # Draw the target course
    destination = unit(DESTINATION_VECTOR) * 10
    draw_line_3D(ax, ORIGIN, destination, 'g')

    # Calculate the D60
    points = truncated_icosahedron(OBSERVED_REFERENCE_BASIS)

    plot_points_3D(ax, points, "", 8.0)

    adjacent_points = find_adjacent_points(points)

    for point, neighbours in enumerate(adjacent_points):
        for i, neighbour in enumerate(neighbours):
            draw_line_3D(ax, points[point], points[neighbour], 'r' if i == 2 else 'b')

    adjacent_glyphs = read_glyph_data()

    # Map the glyphs to points
    glyph_points = associate_glyph_points(adjacent_glyphs, points, adjacent_points)

    # Reorder the list of vectors based on the glyph/point association
    vectors = points[glyph_points]

    # Plot the known star points in magenta
    plot_points_3D(ax, STARMAPPING[np.where(np.any(STARMAPPING != 0, axis=1))], '*m', 16)

    # Plot the inferred star points in cyan
    plot_points_3D(ax, vectors[np.where(np.all(STARMAPPING == 0, axis=1))], '*c', 16)

    # Label all star points
    for glyph in range(n):
        ax.text(vectors[glyph][0], vectors[glyph][1], vectors[glyph][2], GLYPH_NAMES[glyph])

    # Find the closest vector to the target vector
    dot_products = np.zeros(n)

    for i, vector in enumerate(vectors):
        dot_products[i] = np.dot(vector, DESTINATION_VECTOR)

    initial_vector = np.argmax(dot_products)

    # Draw the initial vector
    draw_line_3D(ax, ORIGIN, unit(vectors[initial_vector])*10, 'g--')

    # Dump the data for the spreadsheet
    with open('output.dat', 'w+') as fout:
        for glyph in range(n):
            output_list = list(adjacent_glyphs[glyph]) 
            output_list.append(GLYPH_NAMES[glyph])
            output_list += list(vectors[glyph])
            fout.write(",".join([str(x) for x in output_list])+"\n")

    # Shift back to 1-indexed to match the glyph icons
    print("Initial Vector: Glyph {} : {} : {}".format(initial_vector+1, GLYPH_NAMES[initial_vector], vectors[initial_vector]))

    # Plot the edges of the dual triangular face for visualization
    for corner_a, corner_b in zip(CORNER_VECTORS, np.roll(CORNER_VECTORS,1, axis=0)):
        corner_a = unit(corner_a)
        corner_b = unit(corner_b)
        plot_point_3D(ax, corner_a, 'ko', 4)
        draw_line_3D(ax, corner_a, corner_b, 'k:')

    # Calculate the intersection of the target course and the face corresponding to the initial vector glyph
    p0 = unit(CORNER_VECTORS[0])
    target_point = line_plane_intersection(DESTINATION_VECTOR, vectors[initial_vector], ORIGIN, p0)
    target_centeroid = line_plane_intersection(vectors[initial_vector], vectors[initial_vector], ORIGIN, p0)

    plot_point_3D(ax, target_point, "rx", 16)
    
    plot_point_3D(ax, target_centeroid, "bx", 16)

    local_2d_origin = CORNER_VECTORS[0]
    local_2d_right_vector = unit(CORNER_VECTORS[1] - CORNER_VECTORS[0]) # Along the base of the triangle
    local_2d_up_vector = unit(CORNER_VECTORS[2] - target_centeroid) # From the centeroid to the top

    project_points = [
        target_centeroid,
        CORNER_VECTORS[0],
        CORNER_VECTORS[1],
        CORNER_VECTORS[2],
        target_point,
    ]

    projected_points = np.zeros([len(project_points), 2])

    for i, point in enumerate(project_points):
        d = point - local_2d_origin
        u = np.dot(d, local_2d_right_vector)
        v = np.dot(d, local_2d_up_vector)

        projected_points[i] = np.array([u, v])

    # Rescale the points:
    scale = np.linalg.norm(projected_points[2] - projected_points[1])
    projected_points /= scale

    ax2d = fig.add_subplot(1, 2, 2)
    ax2d.axis('equal')
    plot_points_2D(ax2d, projected_points[:4], 'ko', 8)
    plot_point_2D(ax2d, projected_points[4], 'rx', 8)

    for xy in zip(projected_points[:,0], projected_points[:,1]):
        ax2d.annotate(str(xy), xy, textcoords='data')

    # Work out the glyphs for focusing in on the target point
    # The glyphs refine the target to a triangle 1/64th of the original size over 7 iterations
    # My algorithm only refines the triangle to a 1/4th the original, size, so I need 21 iterations
    target_point_2d = projected_points[4]
    triangle_points_2d = projected_points[1:4]

    triangle_centeroids = []

    steps = []

    for i in range(21):
        triangle_centeroid = np.sum(triangle_points_2d,axis=0)/3
        triangle_centeroids.append(triangle_centeroid)
        triangle_points_2d, step = refine_triangular_segment(triangle_points_2d, target_point_2d)
        plot_points_2D(ax2d, triangle_points_2d, '^', 4)
        steps.append(step)

    step_map = ['A', 'B', 'C', 'D']

    print("Pyramid Fine Adjust Steps:")
    for i in range(len(steps)//3):
        for j in range(3):
            print(step_map[steps[i+j]], end='')
        print('')

    triangle_centeroids = np.array(triangle_centeroids)

    ax2d.plot(triangle_centeroids[:,0], triangle_centeroids[:,1], '^-', markersize=4)

    plt.show()

if __name__ == "__main__":
    main()      

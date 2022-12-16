import collections
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

from skspatial.objects import Plane, Line, Point, Points

import warnings
warnings.filterwarnings("ignore")

from d60_solver import D60Solver
from triangle_solver import TriangleSolver
from util import *

ORIGIN = np.array([0, 0, 0], dtype=np.float64)

OBSERVED_REFERENCE_BASIS = [
    np.array([0,            0.1898387955, 0.9818152737]),
    np.array([0.2084778207, 0.7356418278, 0.6444904486]),
    np.array([0.3373248251, 0.3983170027, 0.8529686558]),
]

DESTINATION_VECTOR = np.array([0.3994985285, -0.7633068197, 0.507704269])

CORNER_VECTORS = [
    np.array([1.7533007652899e-07, -0.93417235727245, 0.35682209419822]), # Vector [44,61,61,61,61,61,61,61]
    np.array([0.57735021142964, -0.57735041155694, 0.57735018458227]), # Vector [44,62,62,62,62,62,62,62]
    np.array([0.52573107107952, -0.8506508337159, 1.4848270733249e-07]), # Vector [44,63,63,63,63,63,63,63]
]

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

    # Draw the target course
    destination = unit(DESTINATION_VECTOR) * 10
    draw_line_3D(ax, ORIGIN, destination, 'g')

    ########################################
    # Solve the D60
    ########################################
    d60 = D60Solver(OBSERVED_REFERENCE_BASIS, 'graph.dat')

    d60.plot(ax)

    initial_glyph, initial_vector = d60.get_nearest_point_vector(DESTINATION_VECTOR)

    # Draw the initial vector
    draw_line_3D(ax, ORIGIN, unit(initial_vector)*10, 'g--')

    # Dump the data for the spreadsheet
    d60.write_starmapping_file('output.dat')

    # Shift back to 1-indexed to match the glyph icons
    print("Initial Vector: Glyph {} : {} : {}".format(initial_glyph+1, d60.glyph_names[initial_glyph], initial_vector))

    ########################################
    # Solve the Triangular Fine-Adjust
    ########################################
    # Get the triangular face
    #triangle_3d = normalize(d60.get_glyph_face_triangle(initial_glyph))
    triangle_3d = np.array(CORNER_VECTORS)
    
    # Technically, the corner vectors are the centeroids of the last layer of sub-triangles at the corners, 
    # rather than the vertices of the super triangle. 
    # 
    # To compensate for this, take the vector from the centeroid to that corner, scale it down by 8**7 and 
    # add it to each corner vector 
    triangle_3d_centeroid = np.mean(triangle_3d, axis=0) 

    for i in range(3):
        triangle_3d[i] += (triangle_3d[i] - triangle_3d_centeroid) / (8**7)

    plot_points_3D(ax, triangle_3d, '^', 10)
    plot_point_3D(ax, triangle_3d_centeroid, '^', 10)

    for i, vertex in enumerate(triangle_3d):
        ax.text(vertex[0], vertex[1], vertex[2], i)

    # Transform the points into 2D
    # Form an orthogonal basis using the triangle centeroid and vertices
    local_3d_origin = triangle_3d[0]
    local_3d_triangle = triangle_3d - local_3d_origin
    local_3d_target_point = Plane.from_points(*triangle_3d).intersect_line(Line(direction=DESTINATION_VECTOR, point=ORIGIN)) - local_3d_origin

    plot_point_3D(ax, local_3d_target_point + local_3d_origin, 'rx', 10)

    local_2d_right_vector = unit(local_3d_triangle[1] - local_3d_triangle[0])
    local_2d_up_vector = unit(local_3d_triangle[2] - np.mean(local_3d_triangle, axis=0))

    print(np.dot(local_2d_up_vector, local_2d_right_vector))

    # Transfer the points into planar coordinates:
    points_3d = Points([*local_3d_triangle, local_3d_target_point])
    points_2d = []

    for point in points_3d:
        u = np.dot(point, local_2d_right_vector)
        v = np.dot(point, local_2d_up_vector)
        points_2d.append(np.array([u, v]))

    points_2d = np.array(points_2d)

    local_2d_triangle = points_2d[:3]
    local_2d_target_point = points_2d[3]

    # Plot the initial points
    ax2d = fig.add_subplot(1, 2, 2)
    ax2d.axis('equal')
    plot_points_2D(ax2d, local_2d_triangle, 'ko', 8)
    plot_point_2D(ax2d, local_2d_target_point, 'rx', 8)

    for xy in zip(local_2d_triangle[:,0], local_2d_triangle[:,1]):
        ax2d.annotate(str(xy), xy, textcoords='data')

    triangle_solver = TriangleSolver('pyramid.dat')

    # Get the sequence of glyphs
    NUM_GLYPH_STEPS = 10
    glyph_seq, triangle_seq = triangle_solver.get_sequence_from_point(local_2d_triangle, local_2d_target_point, NUM_GLYPH_STEPS)

    print("Pyramid Fine Adjust Glyphs:")
    print(glyph_seq)

    # Plot the sequence:
    triangle_centeroids = [np.mean(local_2d_triangle, axis=0)]
    for triangle in triangle_seq:
        plot_points_2D(ax2d, triangle, '^', 8)
        triangle_centeroids.append(np.mean(triangle, axis=0))

    triangle_centeroids = np.array(triangle_centeroids)

    ax2d.plot(triangle_centeroids[:,0], triangle_centeroids[:,1], '^-', markersize=4)

    for i in range(NUM_GLYPH_STEPS):
        glyphs = glyph_seq[:i+1] + [64 for _ in range(NUM_GLYPH_STEPS-1-i)]
        point = triangle_centeroids[i]

        # Calculate a spatial vector for the centeroid at this step
        point_3d = unit(point[1] * local_2d_up_vector + point[0] * local_2d_right_vector + local_3d_origin)

        # Report the steps, the spatial vector, and its dot-product with the destination vector
        frmt = "{:>3}"*NUM_GLYPH_STEPS
        print(frmt.format(*glyphs), end="")
        print(" : ", point_3d, " : ", np.dot(point_3d, DESTINATION_VECTOR))

    plt.show()

if __name__ == "__main__":
    main()      

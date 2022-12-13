import numpy as np

PHI = (1 + 5 ** 0.5) / 2

def unit(vec):
    return vec / np.linalg.norm(vec)

def normalize(points):
    return np.array([unit(vec) for vec in list(points)])

def line_plane_intersection(line_vec, norm_vec, line_point, plane_point):
    d = np.dot((plane_point - line_point), norm_vec) / np.dot(line_vec, norm_vec)
    return line_point + line_vec * d

def line_line_intersection(vec_a, point_a, vec_b, point_b):
    #TODO
    pass

def get_plane_normal_vector(points):
    #TODO
    pass

def project_point_into_plane(point, plane_normal_vec, plane_point):
    #TODO
    pass

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

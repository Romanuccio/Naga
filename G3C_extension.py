"""Collection of functions extending Clifford 3D CGA for work with points, point pairs and spheres."""

import clifford.tools.g3c as tools
from clifford.g3c import *
from numpy import sqrt


def inner_commutator(mv1, mv2):
    """Calculates the commutator w.r.t. inner product of two multivectors."""
    return (mv1 | mv2) - (mv2 | mv1)


def decompose_point_pair(point_pair):
    """Decomposes a point pair into the beginning and end-point Q1, Q2 and normalizes them w.r.t cga points."""
    pp_product = (point_pair | point_pair).value[0]
    if pp_product < 0:
        raise ValueError("Imaginary point pair in decompose_point_pair")
    Q1 = (sqrt(pp_product) - point_pair) / (einf | point_pair)
    scalar_eo = Q1 | einf
    if scalar_eo != 0:
        Q1 /= -scalar_eo
    Q2 = -(sqrt(pp_product) + point_pair) / (einf | point_pair)
    scalar_eo = Q2 | einf
    if scalar_eo != 0:
        Q2 /= -scalar_eo
    Q1_sanitised = Q1(1).clean()
    Q2_sanitised = Q2(1).clean()
    return Q1_sanitised, Q2_sanitised


def extract_point_pair_length(point_pair):
    """Calculates the length of a single point pair."""
    line = point_pair ^ einf
    length = 2 * sqrt((point_pair | point_pair).value[0] / (line | line).value[0])
    return length


def extract_unique_points(point_pairs):
    """Decomposes a collection of point pairs into unique points."""
    if len(point_pairs) == 1:
        return decompose_point_pair(point_pairs[0])

    pts = []

    for i, PP in enumerate(point_pairs):
        point1, point2 = decompose_point_pair(PP)

        pts.append(point1)
        if i == len(point_pairs) - 1:
            pts.append(point2)

    return pts


def translator(translation_vector):
    """Creates a translator representing translation by the cga vector t = t1*e1 + t2*e2 + t3*e3."""
    if (
        translation_vector | e1 == 0
        and translation_vector | e2 == 0
        and translation_vector | e3 == 0
    ):
        return blades[""]
    trans = (-0.5 * translation_vector * einf).exp()
    # bivector = bivector/euclidean_norm(bivector)
    return trans


def translator_to_cga_point(cga_point):
    """Creates a translator to an embedded CGA point."""
    return translator(
        cga_point.value[1] * e1 + cga_point.value[2] * e2 + cga_point.value[3] * e3
    )


def rotor(axis, angle):
    """Creates a rotor representing rotation by an angle around an axis."""
    rot = (axis * -angle / 2.0).exp()
    return rot


def euclidean_norm(multiv):
    """Calculates the euclidean norm of a multivector from its coefficients."""
    blade_coeffs_sum = 0
    for blade_coeff in multiv.value:
        blade_coeffs_sum += blade_coeff**2

    return sqrt(blade_coeffs_sum)


def joint_axis(clifford_cga_vector, angle_x, angle_y):
    """Calculates the axis of rotation in a joint of the snake robot."""
    Txy = translator_to_cga_point(clifford_cga_vector)
    Rx_alpha = rotor(e23, angle_x)
    Ry_axis = Rx_alpha * e13 * ~Rx_alpha
    Ry_axis = Ry_axis.normal()
    Ry_alpha = rotor(Ry_axis, angle_y)
    R_alpha = Ry_alpha * Rx_alpha
    axis = Txy * (R_alpha * e12 * ~R_alpha) * ~Txy
    return axis.normal()


def joint_axis_planar(clifford_cga_vector):
    """Calculates the axis of rotation in a joint of the planar snake robot."""
    T = translator_to_cga_point(clifford_cga_vector)
    if T == blades[""]:
        return e12

    axis = T * e12 * (~T)
    return axis.normal()


def point_pair_from_collection(points):
    """Returns sanitised point pairs defined by a collection of successive conformal points."""
    point_pairs = []
    for i in range(len(points) - 1):
        pair = point_pair(points[i], points[i + 1])
        point_pairs.append(pair)
    return point_pairs


def point_pair(A, B):
    """Returns a sanitised point pair defined by conformal points A and B."""
    pair = A ^ B
    sanitised_pair = pair(2).clean()
    return sanitised_pair


def point_pair_centre(point_pairs):
    """Calculates centres of point pairs."""
    centres = []
    for i in range(len(point_pairs)):
        center_point = tools.get_center_from_sphere(point_pairs[i])
        scalar_eo = center_point | einf
        if scalar_eo != 0:
            center_point /= -scalar_eo
        point_sanitised = center_point(1).clean()
        centres.append(point_sanitised)
    return centres


def extract_points_for_scatter(points):
    """Extracts point coordinates for plotting in a scatter plot."""
    xpos = []
    ypos = []
    zpos = []
    for point in points:
        down_point = down(point)
        xpos.append(down_point.value[1])
        ypos.append(down_point.value[2])
        zpos.append(down_point.value[3])

    return (xpos, ypos, zpos)


def sphere_inner(centre, radius):
    """IPNS sphere representation from conformal centre and radius."""
    sphere = centre - (0.5 * radius**2) * einf
    point_sanitised = sphere(1).clean()
    return point_sanitised


def line_from_points(A, B):
    line_sanitised = (A ^ B ^ einf)(3).clean()
    return line_sanitised


def line_from_pair(P):
    pair_sanitised = (P ^ einf)(3).clean()
    return pair_sanitised


def configuration_test(
    x_pos,
    y_pos,
    z_pos,
    theta,
    phi_1,
    phi_2,
    alpha_x,
    alpha_y,
    beta_x,
    beta_y,
    gamma_x,
    gamma_y,
    l,
):
    """Returns a tuple of points and axes representing the snake robot in the given configuration."""
    point1 = up(x_pos * e1 + y_pos * e2 + z_pos * e3)
    L0 = joint_axis(point1, alpha_x, alpha_y)
    R_theta = rotor(L0, theta)

    point2 = up((x_pos + l) * e1 + y_pos * e2 + z_pos * e3)
    point2 = R_theta * point2 * ~R_theta

    point3 = up((x_pos + 2 * l) * e1 + y_pos * e2 + z_pos * e3)
    L1 = joint_axis(point2, beta_x, beta_y)
    R_phi1 = rotor(L1, phi_1)
    point3 = R_phi1 * R_theta * point3 * ~R_theta * ~R_phi1

    point4 = up((x_pos + 3 * l) * e1 + y_pos * e2 + z_pos * e3)
    L2 = joint_axis(point3, gamma_x, gamma_y)
    R_phi2 = rotor(L2, phi_2)
    point4 = R_phi2 * R_phi1 * R_theta * point4 * ~R_theta * ~R_phi1 * ~R_phi2

    points = [point1, point2, point3, point4]
    axes = [L0, L1, L2]
    return (points, axes)


def configuration_test_planar(x_pos, y_pos, z_pos, theta, phi_1, phi_2, l):
    """Returns a tuple of points and axes representing the snake robot in the given planar configuration."""
    point1 = up(x_pos * e1 + y_pos * e2 + z_pos * e3)
    L0 = joint_axis_planar(point1)
    R_theta = rotor(L0, theta)

    point2 = up((x_pos + l) * e1 + y_pos * e2 + z_pos * e3)
    point2 = R_theta * point2 * ~R_theta

    point3 = up((x_pos + 2 * l) * e1 + y_pos * e2 + z_pos * e3)
    L1 = joint_axis_planar(point2)
    R_phi1 = rotor(L1, phi_1)
    point3 = R_phi1 * R_theta * point3 * ~R_theta * ~R_phi1

    point4 = up((x_pos + 3 * l) * e1 + y_pos * e2 + z_pos * e3)
    L2 = joint_axis_planar(point3)
    R_phi2 = rotor(L2, phi_2)
    point4 = R_phi2 * R_phi1 * R_theta * point4 * ~R_theta * ~R_phi1 * ~R_phi2

    points = [point1, point2, point3, point4]
    axes = [L0, L1, L2]
    return (points, axes)

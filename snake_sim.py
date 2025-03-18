import sys

ALGEBRA_MODULES_PATH = r"..\algebras"
if sys.path[1] != ALGEBRA_MODULES_PATH:
    sys.path.insert(1, ALGEBRA_MODULES_PATH)

import G3C_extension as cga
from numpy import pi
import numpy as np
import pandas as pd

import plotly.express as px
import plotly.graph_objects as go


def test_configuration():
    """Returns a collection of point pairs and axes for a test configuration of the 3-link snake mechanism."""
    # planar config test
    l = 1

    x_pos = 0
    y_pos = 0
    z_pos = 0

    theta = pi / 2
    phi_1 = pi / 3
    phi_2 = -pi / 4
    # phi_1 = 0
    # phi_2 = 0

    alpha_x = 0
    alpha_y = 0

    beta_x = 0
    beta_y = 0

    gamma_x = 0
    gamma_y = 0
    # x_pos = 0
    # y_pos = 0
    # z_pos = 0

    # theta = pi / 2
    # phi_1 = pi / 3
    # phi_2 = -pi / 4

    # alpha_x = pi / 5
    # alpha_y = 0

    # beta_x = -pi / 4
    # beta_y = pi / 3

    # gamma_x = 0
    # gamma_y = pi / 8

    points, axes = cga.configuration_test(
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
    )
    PP = cga.point_pair(points)
    return PP, axes


def test_configuration_3d():
    """Returns a collection of point pairs and axes for a test configuration of the 3-link snake mechanism."""
    # planar config test
    l = 1

    x_pos = 0
    y_pos = 0
    z_pos = 0

    theta = pi / 2
    phi_1 = pi / 3
    phi_2 = -pi / 4

    alpha_x = pi / 5
    alpha_y = 0

    beta_x = -pi / 4
    beta_y = pi / 3

    gamma_x = 0
    gamma_y = pi / 8

    points, axes = cga.configuration_test(
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
    )
    PP = cga.point_pair(points)
    return PP, axes


def simulation_step(PP, axes, delta, linear_combination, time):
    """Calculates a step in the numerical simulation based on the Euler scheme.

    Args:
        PP (list): Collection of point pairs.
        axes (list): Collection of axes.
        delta (float): Time step.
        linear_combination (tuple): Coefficients for the linear combination of solved linsys.
        time (float): Time.

    Returns:
        tuple: Updated collection of point pairs and axes.
    """
    centres = cga.centre(PP)

    # differentials for nonholonomic eq
    dx1 = cga.inner_commutator(centres[0], (cga.e1 * cga.einf))
    dy1 = cga.inner_commutator(centres[0], (cga.e2 * cga.einf))
    dz1 = cga.inner_commutator(centres[0], (cga.e3 * cga.einf))
    dtheta1 = centres[0] | axes[0]

    dp1 = [dx1, dy1, dz1, dtheta1, 0, 0]

    dx2 = cga.inner_commutator(centres[1], (cga.e1 * cga.einf))
    dy2 = cga.inner_commutator(centres[1], (cga.e2 * cga.einf))
    dz2 = cga.inner_commutator(centres[1], (cga.e3 * cga.einf))
    dtheta2 = centres[1] | axes[0]
    dphi1_2 = centres[1] | axes[1]

    dp2 = [dx2, dy2, dz2, dtheta2, dphi1_2, 0]

    dx3 = cga.inner_commutator(centres[2], (cga.e1 * cga.einf))
    dy3 = cga.inner_commutator(centres[2], (cga.e2 * cga.einf))
    dz3 = cga.inner_commutator(centres[2], (cga.e3 * cga.einf))
    dtheta3 = centres[2] | axes[0]
    dphi1_3 = centres[2] | axes[1]
    dphi2_3 = centres[2] | axes[2]

    dp3 = [dx3, dy3, dz3, dtheta3, dphi1_3, dphi2_3]

    dp = [dp1, dp2, dp3]

    # nonholonomic eqs
    equations = []
    for i in range(3):
        equation = [dif ^ PP[i] ^ cga.einf for dif in dp[i]]
        equations.append(equation)

    vectors = [cga.e1, cga.e2, cga.e3]
    planar_systems = []
    # wedge by e1, e2, e3
    for vector in vectors:
        wedged_equations = []
        for equation in equations:
            # extract coeff
            wedged_equation = [
                # TODO change to just taking the mv.value instead of inner product
                (equation[i] ^ vector) | cga.e12345
                for i in range(len(equation))
            ]
            wedged_equations.append(wedged_equation)
        planar_systems.append(wedged_equations)

    linsys = np.array(planar_systems).astype(float)
    # dphi1 = 1
    # dphi2 = 1
    # dphi1 = 0.6*np.sin(time)
    # dphi2 = 0.6*np.sin(time + 2*np.pi/3)
    dphi1 = 1
    dphi2 = 0
    # dphi1 =
    # dphi2 =
    # dphi2 = 0.9*np.cos(3.*time)
    # TODO change to something meaningful after figuring it out
    phi1eq = np.array([0, 0, 0, 1, 0])
    phi2eq = np.array([0, 0, 0, 0, 1])

    rhs = np.array([0, 0, 0, dphi1, dphi2])
    reduced_systems = []
    for i, sys in enumerate(linsys):
        sys = np.delete(sys, i, 1)
        reduced_systems.append(sys)

    sys1 = np.vstack([reduced_systems[0], phi1eq, phi2eq])
    sys2 = np.vstack([reduced_systems[1], phi1eq, phi2eq])
    sys3 = np.vstack([reduced_systems[2], phi1eq, phi2eq])

    reduced_systems = [sys1, sys2, sys3]

    sols = []

    # TODO add check to see that we didn't skip all systems
    for i, sys in enumerate(reduced_systems):
        if np.linalg.det(sys) == 0:
            sols.append(np.zeros(5))
            continue
        sol = np.linalg.solve(sys, rhs)
        sols.append(sol)

    # add complementary dimension back
    # TODO make sure that the zero is inserted at the right place
    expanded_sols = [np.insert(sol, i, 0) for i, sol in enumerate(sols)]

    # solaveragestrange = sum(expanded_sols)
    solaveragestrange = 0
    for i, expanded_sol in enumerate(expanded_sols):
        solaveragestrange += linear_combination[i] * expanded_sol
    solaveragestrange *= delta

    T = cga.translator(
        solaveragestrange[0] * cga.e1
        + solaveragestrange[1] * cga.e2
        + solaveragestrange[2] * cga.e3
    )
    R_theta = cga.rotor(axes[0], solaveragestrange[3])
    R_phi1 = cga.rotor(axes[1], solaveragestrange[4])
    R_phi2 = cga.rotor(axes[2], solaveragestrange[5])

    # rotations = [R_phi2, R_phi1, R_theta]
    rotations = [R_theta, R_phi1, R_phi2]

    moved_PPs = []
    moved_axes = []
    for index, point_pair in enumerate(PP):
        moved_PP = point_pair
        moved_axis = axes[index]
        # TODO should be fine but better check
        for rotation in rotations[0 : (index + 1)]:
            # for rotation in rotations[(len(PP) - index) : len(rotations)]:
            moved_PP = rotation * moved_PP * ~rotation
            moved_axis = rotation * moved_axis * ~rotation
        moved_PP = T * moved_PP * ~T
        # moved_PP = cga.bruteforce_rounding(moved_PP)
        moved_axis = T * moved_axis * ~T
        # moved_axis = cga.bruteforce_rounding(moved_axis)
        moved_axis /= cga.euclidean_norm(moved_axis)
        moved_PPs.append(moved_PP)
        moved_axes.append(moved_axis)

    return moved_PPs, moved_axes


def simulation_run_recursive(
    PP,
    axes,
    delta=0.01,
    steps=300,
    configurations=None,
    configuration_axes=None,
    linear_combination=(1, 1, 1),
):
    """Numerically simulates the snake mechanism with constant inputs $\phi_1=1$, $\phi_2=2$. Uses a recursive approach."""
    if configurations is None:
        configurations = [PP]
    if configuration_axes is None:
        configuration_axes = [axes]

    new_PP, new_axes = simulation_step(
        PP=PP, axes=axes, delta=delta, linear_combination=linear_combination
    )
    configurations.append(new_PP)
    configuration_axes.append(new_axes)

    if steps == 1:
        return configurations, configuration_axes

    return simulation_run_recursive(
        PP=new_PP,
        axes=new_axes,
        delta=delta,
        steps=steps - 1,
        configurations=configurations,
        configuration_axes=configuration_axes,
        linear_combination=linear_combination,
    )


def simulation_run(
    PP,
    axes,
    delta=0.01,
    steps=300,
    configurations=None,
    configuration_axes=None,
    linear_combination=(1, 1, 1),
):
    """Numerically simulates the snake mechanism with constant inputs $\phi_1=1$, $\phi_2=2$."""
    if configurations is None:
        configurations = [PP]
    if configuration_axes is None:
        configuration_axes = [axes]

    time = 0.0

    for _ in range(steps):
        new_PP, new_axes = simulation_step(
            PP=configurations[-1],
            axes=configuration_axes[-1],
            delta=delta,
            linear_combination=linear_combination,
            time=time,
        )
        time += delta
        configurations.append(new_PP)
        configuration_axes.append(new_axes)

    return configurations, configuration_axes


def visualise_simulation_animation(configs_PP, frame_duration=50):
    configurations_dictionary = {
        0: cga.extract_points_for_scatter(cga.extract_unique_points(configs_PP[0]))
    }
    for i, config in enumerate(configs_PP, 1):
        iteration_points = cga.extract_unique_points(config)
        points_coordinates = cga.extract_points_for_scatter(iteration_points)
        configurations_dictionary[i] = points_coordinates

    configs_dataframe = pd.DataFrame(data=configurations_dictionary)
    first_pos = configs_dataframe[configs_dataframe.columns[0]]
    last_pos = configs_dataframe[configs_dataframe.columns[-1]]
    fig = go.Figure(
        data=[
            go.Scatter3d(x=first_pos[0], y=first_pos[1], z=first_pos[2]),
            go.Scatter3d(x=last_pos[0], y=last_pos[1], z=last_pos[2]),
        ],
        layout=go.Layout(
            updatemenus=[
                dict(
                    type="buttons",
                    buttons=[
                        dict(
                            args=[
                                None,
                                {
                                    "frame": {
                                        "duration": frame_duration,
                                        "redraw": True,
                                    },
                                    "fromcurrent": True,
                                },
                            ],
                            label="Play",
                            method="animate",
                        )
                    ],
                )
            ],
            scene={
                "xaxis": dict(range=[-3, 3]),
                "yaxis": dict(range=[-3, 3]),
                "zaxis": dict(range=[-3, 3]),
                "aspectmode": "cube",
            },
            width=700,
            height=700,
        ),
        frames=[
            go.Frame(
                data=[
                    go.Scatter3d(
                        x=configs_dataframe[k][0],
                        y=configs_dataframe[k][1],
                        z=configs_dataframe[k][2],
                    )
                ]
            )
            for k in range(1, len(configurations_dictionary))
        ],
    ).update_traces(
        marker=dict(
            size=3
            )
        )

    fig.show()


def visualise_simulation_start_to_finish(PP_configuration, axes=None):
    initial_PP = PP_configuration[0]
    initial_points = cga.extract_unique_points(initial_PP)
    final_PP = PP_configuration[-1]
    x_initial, y_initial, z_initial = cga.extract_points_for_scatter(initial_points)
    moved_points = cga.extract_unique_points(final_PP)
    x_final, y_final, z_final = cga.extract_points_for_scatter(moved_points)

    df = pd.DataFrame(
        dict(
            X=x_initial + x_final,
            Y=y_initial + y_final,
            Z=z_initial + z_final,
            color=["Initial"] * len(x_initial) + ["Final"] * len(y_final),
        )
    )
    fig = px.line_3d(
        df,
        x="X",
        y="Y",
        z="Z",
        color="color",
        markers=True,
        range_x=(-4, 4),
        range_y=(-4, 4),
        range_z=(-4, 4),
    ).update_layout(
        scene={
            "xaxis": dict(range=[-4, 4]),
            "yaxis": dict(range=[-4, 4]),
            "zaxis": dict(range=[-4, 4]),
            "aspectmode": "cube",
        },
        scene_aspectmode="cube",
        width=700,
        height=700,
    ).update_traces(
        marker=dict(
            size=3
            )
        )  # .update_layout(scene=dict(aspectmode='data'))
    fig.show()


def visualise_PP_configuration(PP):
    initial_points = cga.extract_unique_points(PP)
    x_initial, y_initial, z_initial = cga.extract_points_for_scatter(initial_points)

    df = pd.DataFrame(
        dict(X=x_initial, Y=y_initial, Z=z_initial, color=["Initial"] * len(x_initial))
    )
    fig = px.line_3d(
        df,
        x="X",
        y="Y",
        z="Z",
        color="color",
        markers=True,
        range_x=(-4, 4),
        range_y=(-4, 4),
        range_z=(-4, 4),
    ).update_layout(
        scene={
            "xaxis": dict(nticks=9, range=[-4, 4]),
            "yaxis": dict(nticks=9, range=[-4, 4]),
            "zaxis": dict(nticks=9, range=[-4, 4]),
            "aspectmode": "cube",
        },
        scene_aspectmode="cube",
        width=700,
        height=700,
    ).update_traces(
        marker=dict(
            size=3
            )
        )  # .update_layout(scene=dict(aspectmode='data'))
    fig.show()


# def configuration_multilink(x0, y0, z0=0):
#     """Takes lists of initial points and returns them wedged as point pairs."""


def configuration_multilink_random_planar(count=10, length=0.3):
    count += 1
    pts = [None] * count
    pts[0] = cga.up(0)
    for i in range(1, count):
        angle = np.clip(np.random.uniform() * np.pi / 2.0, np.pi / 36.0, np.pi / 2.0)
        translation = cga.translator_to_cga_point(cga.up(length * (np.cos(angle) * cga.e1 + np.sin(angle) * cga.e2)))
        new_point = translation * pts[i - 1] * ~translation
        pts[i] = new_point

    return cga.point_pair(pts)


def configuration_multilink_line(count=10, length=0.3):
    count += 1
    pts = [None] * count
    pts[0] = cga.up(0)
    for i in range(1, count):
        translation = cga.translator_to_cga_point(cga.up(length * (cga.e1 + cga.e2)))
        new_point = translation * pts[i - 1] * ~translation
        pts[i] = new_point

    return cga.point_pair(pts)


def fakesnake(initial_PP, xdot, ydot, zdot):
    PP_length = cga.extract_point_pair_length(initial_PP[0])
    # translate
    Txyz = cga.translator(xdot*cga.e1 + ydot*cga.e2 + zdot*cga.e3)
    # translate all point pairs
    new_PP = [Txyz*pair*~Txyz for pair in initial_PP]
    
    for i, new_pair in enumerate(new_PP):
        # initial configuration points A, B
        _, initial_B = cga.decompose_point_pair(initial_PP[i])
        _, unrotated_B = cga.decompose_point_pair(new_PP[i])

        # line passing through initial point pair determining nonholonomic cond
        line_PP = (initial_PP[i]^cga.einf) #.normal()

        # decompose new translated first point pair
        new_A, _ = cga.decompose_point_pair(new_pair)

        # second point of moved point pair must lay on line passing through initial position
        sphere = new_A - 0.5*PP_length**2*cga.einf
        intersection = line_PP.normal() & sphere.dual()

        # take point closer to initial B
        new_B1, new_B2 = cga.decompose_point_pair(intersection)

        # TODO fix this shit
        new_B = new_B1 if cga.point_distance(initial_B, new_B1) < cga.point_distance(initial_B, new_B2) else new_B2
        
        # measure change of angle
        # inner product of line passing through original point_pair and line passing through new point pair
        new_pair = new_A^new_B
        new_line = (new_pair^cga.einf).normal()
        product = line_PP.normal()|new_line
        dangle = np.arccos(product.value[0])
        # if np.isclose(dangle, 0):
        #     print("wut")

        # the new plane of rotation is the one containing the translated point pair and the original point pair
        plane = new_pair^unrotated_B^cga.einf
        plane_normal = (plane.dual()|((cga.e1^cga.e2^cga.e3).inv()))*(cga.e1^cga.e2^cga.e3)
        plane_normal = plane_normal.normal()
        # axis = (cga.up(plane_normal)^cga.up(2*plane_normal)) #.normal()
        axis = cga.up(plane_normal)^cga.up(0)
        
        # translate axis to new_A
        T_new_A = cga.translator_to_cga_point(new_A)
        axis = (T_new_A*axis*~T_new_A)
        # axis /= cga.euclidean_norm(axis)
        axis = axis.normal()

        # R_angle = cga.rotor(axis=(axis^cga.einf).dual(), angle=-dangle)
        R_angle = cga.rotor(axis=(axis^cga.einf).dual(), angle=((-1)**(i+1))*dangle)
        new_PP[i] = new_pair
        
        if i == len(initial_PP) - 1:
            break
        
        for j in range(i + 1, len(new_PP)):
            new_PP[j] = R_angle*new_PP[j]*~R_angle
    
    return new_PP


def upgraded_fakesnake(initial_PP, xdot, ydot, zdot, dt):
    PP_length = cga.extract_point_pair_length(initial_PP[0])
    initial_centres = cga.centre(initial_PP)

    # direction of first PP line
    line_direction = (initial_PP[0]^cga.einf)|(cga.einf^cga.eo)
    # move centre in direction of line a bit
    first_centre = initial_centres[0]
    centre_movement_vector = -line_direction*dt
    T = cga.translator(centre_movement_vector)
    new_centre = T*first_centre*~T

    A, _ = cga.decompose_point_pair(initial_PP[0])
    # sphere of radius 1/2 of link length around new centre
    sphere = cga.sphere_inner(new_centre, PP_length/2.)

    # control direction
    control_direction = xdot*cga.e1 + ydot*cga.e2 + zdot*cga.e3
    control_direction_translator = cga.translator(control_direction)
    # translate initial A in direction of steering input
    steering_point = control_direction_translator*A*~control_direction_translator
    # find new A by intersecting it with the new centre sphere and the steering line
    steering_line = A^steering_point^cga.einf
    intersection = steering_line & sphere.dual()
    # choose one of intersection points to control
    new_A, new_B = cga.decompose_point_pair(intersection)
    # choose the point closer to the initial steering point
    distance_new_A = cga.point_distance(steering_point, new_A)
    distance_new_B = cga.point_distance(steering_point, new_B)

    if distance_new_A < distance_new_B:
        new_point = new_A
    else:
        new_point = new_B
    # translate
    Txyz = cga.translator(xdot*cga.e1 + ydot*cga.e2 + zdot*cga.e3)
    # translate all point pairs
    new_PP = [Txyz*pair*~Txyz for pair in initial_PP]
    
    for i, new_pair in enumerate(new_PP):
        # initial configuration points A, B
        _, initial_B = cga.decompose_point_pair(initial_PP[i])
        _, unrotated_B = cga.decompose_point_pair(new_PP[i])

        # line passing through initial point pair determining nonholonomic cond
        line_PP = (initial_PP[i]^cga.einf) #.normal()

        # decompose new translated first point pair
        new_A, _ = cga.decompose_point_pair(new_pair)

        # second point of moved point pair must lay on line passing through initial position
        sphere = new_A - 0.5*PP_length**2*cga.einf
        intersection = line_PP.normal() & sphere.dual()

        # take point closer to initial B
        new_B1, new_B2 = cga.decompose_point_pair(intersection)

        new_B = new_B1 if cga.point_distance(initial_B, new_B1) < cga.point_distance(initial_B, new_B2) else new_B2
        
        # measure change of angle
        # inner product of line passing through original point_pair and line passing through new point pair
        new_pair = new_A^new_B
        new_line = (new_pair^cga.einf).normal()
        product = line_PP.normal()|new_line
        dangle = np.arccos(product.value[0])
        # if np.isclose(dangle, 0):
        #     print("wut")

        # the new plane of rotation is the one containing the translated point pair and the original point pair
        plane = new_pair^unrotated_B^cga.einf
        plane_normal = (plane.dual()|((cga.e1^cga.e2^cga.e3).inv()))*(cga.e1^cga.e2^cga.e3)
        plane_normal = plane_normal.normal()
        # axis = (cga.up(plane_normal)^cga.up(2*plane_normal)) #.normal()
        axis = cga.up(plane_normal)^cga.up(0)
        
        # translate axis to new_A
        T_new_A = cga.translator_to_cga_point(new_A)
        axis = (T_new_A*axis*~T_new_A)
        # axis /= cga.euclidean_norm(axis)
        axis = axis.normal()

        # R_angle = cga.rotor(axis=(axis^cga.einf).dual(), angle=-dangle)
        R_angle = cga.rotor(axis=(axis^cga.einf).dual(), angle=((-1)**(i+1))*dangle)
        new_PP[i] = new_pair
        
        if i == len(initial_PP) - 1:
            break
        
        for j in range(i + 1, len(new_PP)):
            new_PP[j] = R_angle*new_PP[j]*~R_angle
    
    return new_PP

"""Module containing functions for 3D plotly-based visualisations and generation of snake robot configurations."""

import G3C_extension as cga
from G3C_extension import e1, e2, e3, einf
import clifford.tools.g3c as tools
from numpy import pi
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

def gigafakesnake(initial_PP, xdot, ydot, zdot, initial_PP_length, eps=10**(-3)):
    """Calculates one step of the kinematics algorithm."""
    new_PP = [None] * len(initial_PP)
    # step 1: find new possible configuration
    
    def point_pair_step(new_A, initial_B, initial_line):
        ###### b) construct sphere centered on new_A of radius half initial_PP_length
        centre_range_sphere = cga.sphere_inner(new_A, 0.5*initial_PP_length)
        ###### c) find new possible centres by intersection of sphere with initial line
        # TODO zeptat se petra na grade dvojbodu
        intersection = initial_line & centre_range_sphere.dual()
        intersection = intersection.clean()
        ###### d) check if intersection is real, decompose
        if (intersection|intersection).value[0] < 0:
                raise ValueError('infeasible configuration due to imaginary point pair')
            
        # take centre closer to initial B
        # TODO consider different functional
        new_centre_1, new_centre_2 = cga.decompose_point_pair(intersection)
        new_centre = new_centre_1 if tools.euc_dist(initial_B, new_centre_1) < tools.euc_dist(initial_B, new_centre_2) else new_centre_2
        
        ###### e) find new point B as intersection of line (new_A through new_centre) and sphere centered on new_A
        new_line = cga.line_from_points(new_A, new_centre)
        link_length_sphere = cga.sphere_inner(new_A, initial_PP_length)
        intersection = new_line & link_length_sphere.dual()
        intersection = intersection.clean()
        new_B_1, new_B_2 = cga.decompose_point_pair(intersection)
        new_B = new_B_1 if tools.euc_dist(initial_B, new_B_1) < tools.euc_dist(initial_B, new_B_2) else new_B_2
        return new_B

        
    ###### a) prepare everything needed: lines passing through initial point pairs, starting points
    initial_lines = [cga.line_from_pair(point_pair) for point_pair in initial_PP]
    new_A = None
    
    ###### for every point pair in last configuration, apply the IK algorithm
    for i, PP in enumerate(initial_PP):
        initial_A, initial_B = cga.decompose_point_pair(PP)
        if new_A is None:
            # first step in algorithm
            # controlled head: translate as desired to obtain new head position
            Txyz = cga.translator(xdot*e1 + ydot*e2 + zdot*e3)
            new_A = Txyz*initial_A*~Txyz
            new_A = new_A.clean()
        
        # find new location of second point in point pair
        new_B = point_pair_step(new_A, initial_B, initial_lines[i])
        # construct new point pair
        new_pair = new_A^new_B
        new_PP[i] = new_pair.clean()
        # set the new position endpoint as the moved head point for next point pair
        new_A = new_B
        
    ###### checking for numerical stability: length of point pairs should not change
    # if the length changes, it is usually caused by ghost blades
    for PP in new_PP:
        new_length = cga.extract_point_pair_length(PP)
        if np.abs(initial_PP_length - new_length) > eps:
            print(f'initial: {initial_PP_length}, new: {new_length}')
            # TODO remove this ffs
            raise ValueError('pokroutil se mi had :(')
    
    return new_PP


def calculate_kinematics(initial_PP_configuration, dx, dy, dz, iterations=100, eps=10**(-3)):
    """Calculates successive configurations for given amount of iterations using the kinematics algorithm."""
    initial_PP_length = cga.extract_point_pair_length(initial_PP_configuration[0])
    new_PP = [None] * (iterations + 1)
    new_PP[0] = initial_PP_configuration
    for i in range(iterations):
        new_PP[i + 1] = gigafakesnake(new_PP[i], xdot=dx[i], ydot=dy[i], zdot=dz[i], initial_PP_length=initial_PP_length, eps=eps)
    return new_PP


def visualise_simulation_animation(configs_PP, frame_duration=50):
    """Plots the 3D animation of the snake robot."""
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
                "xaxis": dict(range=[-3, 3], fixedrange=False),
                "yaxis": dict(range=[-3, 3], fixedrange=False),
                "zaxis": dict(range=[-3, 3], fixedrange=False),
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
    
    
    def visualise_simulation_animation_traces(configs_PP, frame_duration=50):
    """Plots the 3D animation of the snake robot."""
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


def visualise_simulation_start_to_finish(PP_configuration):
    """Plots the initial and final position of a list of point pairs in 3D."""
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
        )
    fig.show()


def visualise_simulation_evolution(PP_configuration, link_count, range_x = None, range_y = None, range_z = None, color_disc_map = None):
    """Plots the evolution of positions of a list of point pairs in 3D."""
    x = []
    y = []
    z = []
    if range_x is None:
        range_x = (-4, 4)
    if range_y is None:
        range_y = (-4, 4)
    if range_z is None:
        range_z = (-4, 4)
    if color_disc_map is None:
        # color_disc_map = {val: f'rgba({255*val}, 0, {255*(1.-val)}, 1)' for val in df.color}
        color_disc_map = lambda val: f'rgba({255*val}, 0, {255*(1.-val)}, 1)'
        
    for configuration in PP_configuration:
        points = cga.extract_unique_points(configuration)
        xn, yn, zn = cga.extract_points_for_scatter(points)
        x += xn
        y += yn
        z += zn
    colors = np.repeat(np.linspace(0, 1, len(PP_configuration)), link_count+1)
    df = pd.DataFrame(
        dict(
            X=x,
            Y=y,
            Z=z,
            color=colors
        )
    )
    fig = px.line_3d(
        df,
        x="X",
        y="Y",
        z="Z",
        color="color",
        # color_discrete_map={val: f'rgba(0, 0, 255, {val*0.3 + 0.7})' for val in df.color},
        color_discrete_map= {col: color_disc_map(col) for col in df.color},
        # color_discrete_map={val: f'rgb({147*(.3*val+.7)}, {173*(.3*val+.7)}, {68*(.3*val+.7)})' for val in df.color},
        markers=True,
        range_x=range_x,
        range_y=range_y,
        range_z=range_z,
    ).update_layout(
        scene={
            "xaxis": dict(range=range_x),
            "yaxis": dict(range=range_y),
            "zaxis": dict(range=range_z),
            # "aspectmode": "cube",
        },
        scene_aspectmode="data",
        width=700,
        height=700,
    ).update_traces(
        marker=dict(
            size=3
            ),
        # opacity=[0., .2, .3, .4, .5]
        )
    fig.show()

def visualise_PP_configuration(PP):
    """Plots one configuration of a list of point pairs in 3D."""
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
        )
    fig.show()



def configuration_multilink_random_planar(count=10, length=0.3):
    """Generates a planar configuration of count point pairs rotated at random angles."""
    count += 1
    pts = [None] * count
    pts[0] = cga.up(0)
    for i in range(1, count):
        angle = np.clip(np.random.uniform() * np.pi / 2.0, np.pi / 36.0, np.pi / 2.0)
        translation = cga.translator_to_cga_point(cga.up(length * (np.cos(angle) * cga.e1 + np.sin(angle) * cga.e2)))
        new_point = translation * pts[i - 1] * ~translation
        pts[i] = new_point

    return cga.point_pair_from_collection(pts)


def configuration_multilink_line(count=10, length=0.3):
    """Generates a configuration of count connected point pairs of length length in a line."""
    count += 1
    pts = [None] * count
    pts[0] = cga.up(0)
    translation = cga.translator_to_cga_point(cga.up(length/np.sqrt(2) * (cga.e1 + cga.e2)))
    for i in range(1, count):
        new_point = translation * pts[i - 1] * ~translation
        pts[i] = new_point

    return cga.point_pair_from_collection(pts)
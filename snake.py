import G3C_extension as cga
from G3C_extension import e1, e2, e3, einf
import numpy as np

def gigafakesnake(initial_PP, xdot, ydot, zdot, initial_PP_length, eps=10**(-3)):
    """Calculates one step of the kinematics algorithm."""
    new_PP = [None] * len(initial_PP)
    # step 1: find new possible configuration
    
    def point_pair_step(new_A, initial_B, initial_line):
        ###### b) construct sphere centered on new_A of radius half initial_PP_length
        centre_range_sphere = new_A - 0.5*(0.5*initial_PP_length)**2*einf
        
        ###### c) find new possible centres by intersection of sphere with initial line
        intersection = initial_line & centre_range_sphere.dual()
        
        ###### d) check if intersection is real, decompose
        if (intersection|intersection).value[0] < 0:
                raise ValueError('infeasible configuration')
            
        # take centre closer to initial B
        # TODO consider different functional
        new_centre_1, new_centre_2 = cga.decompose_point_pair(intersection)
        new_centre = new_centre_1 if cga.point_distance(initial_B, new_centre_1) < cga.point_distance(initial_B, new_centre_2) else new_centre_2
        
        ###### e) find new point B as intersection of line (new_A through new_centre) and sphere centered on new_A
        new_line = new_A^new_centre^einf
        link_length_sphere = new_A - 0.5*initial_PP_length**2*einf
        intersection = new_line & link_length_sphere.dual()
        new_B_1, new_B_2 = cga.decompose_point_pair(intersection)
        new_B = new_B_1 if cga.point_distance(initial_B, new_B_1) < cga.point_distance(initial_B, new_B_2) else new_B_2
        return new_B
        
    ###### a) prepare everything needed: lines passing through initial point pairs, starting points
    initial_lines = [(point_pair^einf).normal() for point_pair in initial_PP]
    new_A = None
    
    for i, PP in enumerate(initial_PP):
        initial_A, initial_B = cga.decompose_point_pair(PP)
        if new_A is None:
            # controlled head: translate as desired to obtain new head position
            Txyz = cga.translator(xdot*e1 + ydot*e2 + zdot*e3)
            new_A = Txyz*initial_A*~Txyz
            
        new_B = point_pair_step(new_A, initial_B, initial_lines[i])
        new_pair = new_A^new_B
        new_PP[i] = new_pair
        new_A = new_B
        
    for PP in new_PP:
        new_length = cga.extract_point_pair_length(PP)
        if np.abs(initial_PP_length - new_length) > eps:
            raise ValueError('zkratil se mi had :(')
    
    return new_PP
import sys
from numpy import cos, sin, pi, sqrt, e
import numpy.linalg as npl
import numpy as np
import G3C_extension as cga
from G3C_extension import e1, e2, e3, eo, einf, up
import pytest
import copy


class Test_PP:
    @pytest.fixture(autouse=True)
    def initialisation(self):
        """Initialisation of a point pair represented by points A, B"""
        # self.A_coord = [1, 1, -1]
        # self.B_coord = [-1, -2, 3]
        self.A_coord = [1, 2, 3]
        self.B_coord = [-1, -3, -1]
        self.orig_PP_length = sqrt(
            (self.A_coord[0] - self.B_coord[0]) ** 2
            + (self.A_coord[1] - self.B_coord[1]) ** 2
            + (self.A_coord[2] - self.B_coord[2]) ** 2
        )
        self.A = up(self.A_coord[0] * e1 + self.A_coord[1] * e2 + self.A_coord[2] * e3)
        self.B = up(self.B_coord[0] * e1 + self.B_coord[1] * e2 + self.B_coord[2] * e3)
        scalar_eo = self.A | einf
        if scalar_eo != 0:
            self.A /= self.A | einf
        scalar_eo = self.B | einf
        if scalar_eo != 0:
            self.B /= self.B | einf

        self.orig_PP = self.A ^ self.B

    def test_PP_creation(self):
        """Test method for G3C point pair creation."""
        PP = cga.point_pair(self.A, self.B)
        assert self.orig_PP == PP

    def test_PP_decomposition_first_point(self):
        """Test for point pair decomposition for first point."""
        C, D = cga.decompose_point_pair(self.orig_PP)
        assert self.A + C == cga.blades[""] * 0

    def test_PP_decomposition_second_point(self):
        """Test for point pair decomposition for second point."""
        C, D = cga.decompose_point_pair(self.orig_PP)
        assert self.B + D == cga.blades[""] * 0

    def test_PP_length(self):
        """Determines if G3C euclidean norm is correct."""
        length = cga.euclidean_norm(cga.down(self.A) - cga.down(self.B))
        assert np.isclose(length, self.orig_PP_length)

    def test_PP_length_by_IP(self):
        """Determines if PP length calculated by inner product is equal to real length."""
        extraction = (einf | self.orig_PP) + ((einf | self.orig_PP) | eo) * einf
        length = cga.euclidean_norm(extraction)
        assert length == self.orig_PP_length

    def test_rotated_PP_length(self):
        """Tests if length of PP changes by applying the same rotor consecutively."""
        R = cga.rotor(cga.e12, pi / 3.0)
        rotated_PP = copy.deepcopy(self.orig_PP)
        C, D = cga.decompose_point_pair(rotated_PP)
        print(f"initial:\nC\n{C.value}\nD\n{D.value}\n")
        for _ in range(300):
            rotated_PP = R * rotated_PP * ~R
        C, D = cga.decompose_point_pair(rotated_PP)
        print(f"300 rotations:\nC\n{C.value}\nD\n{D.value}\n")
        for _ in range(27000):
            rotated_PP = R * rotated_PP * ~R
        C, D = cga.decompose_point_pair(rotated_PP)
        print(f"30000 rotations:\nC\n{C.value}\nD\n{D.value}")
        down_C, down_D = (cga.down(C), cga.down(D))
        vect = down_C - down_D
        length = cga.euclidean_norm(vect)
        # length = sqrt((self.orig_PP | self.orig_PP).value[0])
        difference = self.orig_PP_length - length
        assert np.isclose(difference, 0)

    def test_planarity_of_point_pair_after_transformations(self):
        """Tests if a point pair in the xy-plane remains planar after lots of transformations."""
        A = up(0)
        B = up(e1 + e2)
        C = up(-e1 - e2)
        PP1 = A ^ B
        PP2 = B ^ C
        R = cga.rotor(e1 ^ e2, 0.01)

        for i in range(10_000):
            T = cga.translator(np.sin(i / 100.0) * e1 + np.sin(i / 100.0) * e2)
            PP1 = R * T * PP1 * ~T * ~R
            PP2 = R * T * PP2 * ~T * ~R

        final_A, final_B1 = cga.decompose_point_pair(PP1)
        final_B2, final_C = cga.decompose_point_pair(PP2)

        for point in [final_A, final_B1, final_B2, final_C]:
            assert point | e3 == 0

    def test_rotor_exponentials(self):
        """Checks if .exp() clifford function returns the same rotor as G3C obtained by np.e**(biv)."""
        # rotor from exponentiation of numpy.e
        axis = cga.e12
        angle = pi / 3.0
        biv = -axis * angle / 2.0
        R1 = cga.rotor(axis, angle)
        R2 = biv.exp()
        print(f"R1: {R1} \n{R1.value}\nR2: {R2} \n{R2.value}")
        assert R1 == R2

    def test_rotor_exp_goniometric_equality(self):
        """Checks if G3C rotor exponential form is the same as goniometric form."""
        axis = cga.e12
        angle = pi / 3.0
        R1 = cga.rotor(axis, angle)
        R2 = cos(angle / 2) - axis * sin(angle / 2)
        assert R1 == R2

    def test_PP_decomposition(self):
        """Checks if decomposing and recomposing PP is stable."""
        PP = copy.deepcopy(self.orig_PP)
        for _ in range(300):
            print(PP)
            A, B = cga.decompose_point_pair(PP)
            print(A, B)
            PP = cga.point_pair(A, B)

        assert PP == pytest.approx(self.orig_PP)

    def test_rotated_PP_exp_goniometric_equality(self):
        """Checks equality of """
        axis = cga.e12
        angle = pi/3.
        R1 = cga.rotor(axis, angle)
        R2 = cos(angle/2) - axis*sin(angle/2)
        assert R1 == R2

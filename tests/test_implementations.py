import numpy as np
from implementations import build_poly

def test_build_poly():
    x = np.array([
        [1, 3],
        [2, 4]
    ])
    poly_x = build_poly(x, degree=3)

    expected_poly_x = np.array([
        [1, 1, 1, 3, 9, 27],
        [2, 4, 8, 4, 16, 64]
    ])

    assert np.array_equal(poly_x, expected_poly_x), "Polynomial features were not generated correctly."

    poly_y = build_poly(x, degree=1)
    assert np.array_equal(poly_y, x), "Polynomial features were not generated correctly."
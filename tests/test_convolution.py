import numpy as np
from pages.convolution import compute_convolution
from utils.math_utils import tri


def test_rect_convolution_rect_results_triangular():
    result = compute_convolution('rect(t)', 'rect(t)')
    t = np.array(result['t'])
    y_conv = np.array(result['y_conv'])
    expected = tri(t)
    assert np.allclose(y_conv, expected, atol=3e-2)
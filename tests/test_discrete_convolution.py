import numpy as np
from pages.discrete_convolution import compute_discrete_convolution

def test_shifted_rect_is_not_zero():
    res = compute_discrete_convolution('rect_3(k-29)', 'rect_3(k+1)')
    y1 = np.array(res['y1'])
    assert np.any(y1 != 0)
    y_conv = np.array(res['y_conv'])
    assert np.any(y_conv != 0)
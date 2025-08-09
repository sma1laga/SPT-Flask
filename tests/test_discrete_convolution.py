import numpy as np
from pages.discrete_convolution import compute_discrete_convolution

def test_shifted_rect_is_not_zero():
    # The implementation now expects the discrete rectangle notation using
    # square brackets, e.g. ``rect_3[k-29]``.  Adapt the test accordingly.
    res = compute_discrete_convolution('rect_3[k-29]', 'rect_3[k+1]')
    y1 = np.array(res['y1'])
    assert np.any(y1 != 0)
    y_conv = np.array(res['y_conv'])
    assert np.any(y_conv != 0)


def test_unbounded_step_convolution_forms_ramp():
    """Convolution of two step sequences should yield an unbounded ramp."""
    res = compute_discrete_convolution('step[k]', 'step[k]')

    # The display axis should remain limited despite the unbounded nature of the
    # sequences, ensuring the plot remains readable.
    k = np.array(res['k'])
    assert k.max() < 100
    assert k.min() > -50

    # The convolution itself should follow the discrete ramp: max(0, n+1)
    k_conv = np.array(res['k_conv'])
    y_conv = np.array(res['y_conv'])
    expected = np.where(k_conv >= 0, k_conv + 1, 0)
    assert np.allclose(y_conv, expected)
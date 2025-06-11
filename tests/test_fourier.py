from pages.fourier_page import compute_fourier


def test_compute_fourier_returns_expected_keys():
    res = compute_fourier('sin(t)', 0.0)
    keys = {'t', 'y_real', 'y_imag', 'f', 'magnitude', 'phase', 'transformation_label'}
    assert keys.issubset(res.keys())
    length = len(res['t'])
    assert all(len(res[k]) == length for k in ['y_real', 'y_imag', 'f', 'magnitude', 'phase'])
    assert max(res['magnitude']) == 1
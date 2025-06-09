import numpy as np
from flask import Blueprint, render_template, request, jsonify

transform_intuition_bp = Blueprint(
    'transform_intuition', __name__,
    template_folder='templates/discrete')


def _generate_data(N=8, r_min=0.5, r_max=1.5, r_points=50, theta_points=160):
    r = np.linspace(r_min, r_max, r_points)
    theta = np.linspace(0, 2*np.pi, theta_points)
    R, Theta = np.meshgrid(r, theta)
    Z = R * np.exp(1j * Theta)
    Xz = np.zeros_like(Z, dtype=complex)
    for n in range(N):
        Xz += Z ** (-n)  # x[n] = 1
    mag = np.abs(Xz)
    if mag.max() > 0:
        mag /= mag.max()
    x_grid = (R * np.cos(Theta)).tolist()
    y_grid = (R * np.sin(Theta)).tolist()
    z_grid = mag.tolist()

    # DTFT on unit circle (r=1)
    theta_line = theta
    exps = np.exp(-1j * np.outer(np.arange(N), theta_line))
    X_line = exps.sum(axis=0)
    line_mag = np.abs(X_line)
    if line_mag.max() > 0:
        line_mag /= line_mag.max()
    dtft_x = np.cos(theta_line).tolist()
    dtft_y = np.sin(theta_line).tolist()
    dtft_z = line_mag.tolist()

    # DFT samples
    theta_dft = np.linspace(0, 2*np.pi, N, endpoint=False)
    exps_dft = np.exp(-1j * np.outer(np.arange(N), theta_dft))
    X_dft = exps_dft.sum(axis=0)
    dft_mag = np.abs(X_dft)
    if dft_mag.max() > 0:
        dft_mag /= dft_mag.max()
    dft_x = np.cos(theta_dft).tolist()
    dft_y = np.sin(theta_dft).tolist()
    dft_z = dft_mag.tolist()

    return {
        'x_grid': x_grid,
        'y_grid': y_grid,
        'z_grid': z_grid,
        'dtft_x': dtft_x,
        'dtft_y': dtft_y,
        'dtft_z': dtft_z,
        'dft_x': dft_x,
        'dft_y': dft_y,
        'dft_z': dft_z,
        'N': N,
    }


@transform_intuition_bp.route('/')
def transform_intuition():
    data = _generate_data()
    return render_template('dft_dtft_z.html', **data)


@transform_intuition_bp.route('/update', methods=['POST'])
def update_transform_intuition():
    data = request.get_json(force=True) or {}
    N = max(1, int(data.get('N', 8)))
    r_min = float(data.get('r_min', 0.5))
    r_max = float(data.get('r_max', 1.5))
    return jsonify(_generate_data(N=N, r_min=r_min, r_max=r_max))

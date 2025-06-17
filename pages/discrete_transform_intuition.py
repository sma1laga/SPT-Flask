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

    x_grid = (R * np.cos(Theta)).tolist()
    y_grid = (R * np.sin(Theta)).tolist()

    # DTFT on unit circle (r=1)
    theta_line = theta
    exps = np.exp(-1j * np.outer(np.arange(N), theta_line))
    X_line = exps.sum(axis=0)
    # ----------------------------------------------------------
    # Use the peak of the DTFT as the global reference magnitude
    unit_max = np.abs(X_line).max()
    if unit_max == 0:
        unit_max = 1.0          # avoid divide-by-zero for the all-zero case
    # ----------------------------------------------------------
    line_mag = np.abs(X_line)
    mag      /= unit_max       # 3-D surface
    line_mag /= unit_max       # DTFT curve
    z_grid = mag.tolist()
    dtft_x = np.cos(theta_line).tolist()
    dtft_y = np.sin(theta_line).tolist()
    dtft_z = line_mag.tolist()

    # DFT samples
    theta_dft = np.linspace(0, 2*np.pi, N, endpoint=False)
    exps_dft = np.exp(-1j * np.outer(np.arange(N), theta_dft))
    X_dft = exps_dft.sum(axis=0)
    dft_mag = np.abs(X_dft)
    dft_mag  /= unit_max       # DFT samples
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

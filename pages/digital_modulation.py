import numpy as np
from flask import Blueprint, render_template, request, jsonify

dig_bp = Blueprint('digital_modulation', __name__, template_folder='templates')

def make_time(t_end=1.0, fs=1000):
    return np.linspace(0, t_end, int(fs*t_end), endpoint=False)

@dig_bp.route('/')
def digital_modulation():
    return render_template('digital_modulation.html')

@dig_bp.route('/api/digital_modulate')
def digital_modulate_api():
    kind = request.args.get('type', 'ASK')
    fc   = float(request.args.get('fc', 100))
    br   = float(request.args.get('br', 5))
    dev  = float(request.args.get('dev', 20))
    t    = make_time()
    bit_period = 1 / br
    # simple NRZ bits: high for first half of each bit period, low next half
    bits = (np.mod(t, bit_period) < (bit_period/2)).astype(int)

    if kind == 'ASK':
        modulated = bits * np.cos(2*np.pi*fc*t)

    elif kind == 'PSK':
        # bit=1 → phase=0; bit=0 → phase=π
        modulated = np.cos(2*np.pi*fc*t + np.pi*(1-bits))

    elif kind == 'FSK':
        # two frequencies f0, f1 around fc
        f0 = fc - dev
        f1 = fc + dev
        modulated = np.cos(2*np.pi*(f0 + (f1-f0)*bits)*t)

    else:
        return jsonify(error="Unknown modulation type"), 400

    return jsonify({
        't':        t.tolist(),
        'bits':     bits.tolist(),
        'modulated': modulated.tolist()
    })

@dig_bp.route('/api/digital_demodulate')
def digital_demodulate_api():
    kind = request.args.get('type', 'ASK')
    # for demo, just regenerate bits and echo as "demodulated"
    t    = make_time()
    br   = float(request.args.get('br', 5))
    bit_period = 1 / br
    bits = (np.mod(t, bit_period) < (bit_period/2)).astype(int)
    # regenerate tx signal (same as modulate)
    # you could implement real demod, but for now we just echo bits
    modulated = np.array(bits)
    demodulated = bits

    return jsonify({
        't':            t.tolist(),
        'modulated':    modulated.tolist(),
        'demodulated':  demodulated.tolist()
    })

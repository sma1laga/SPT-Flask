# modulation.py

import numpy as np
from flask import Blueprint, render_template, request, jsonify

mod_bp = Blueprint('modulation', __name__)

# ---------- helpers ----------

def make_time(t_end=1.0, fs=1000):
    N = int(max(8, fs * t_end))
    return np.linspace(0.0, t_end, N, endpoint=False)

def clamp(x, lo, hi):
    return max(lo, min(hi, x))

def analytic_signal(x: np.ndarray) -> np.ndarray:
    """Analytic signal via frequency-domain Hilbert transform (no SciPy)."""
    N = x.size
    X = np.fft.fft(x)
    h = np.zeros(N)
    h[0] = 1.0
    if N % 2 == 0:
        h[N//2] = 1.0
        h[1:N//2] = 2.0
    else:
        h[1:(N+1)//2] = 2.0
    return np.fft.ifft(X * h)

def spectrum_db(x: np.ndarray, fs: float):
    N = x.size
    win = np.hanning(N)
    X = np.fft.rfft(x * win)
    f = np.fft.rfftfreq(N, d=1.0/fs)
    P = (np.abs(X) ** 2) / (np.sum(win**2))
    P_db = 10*np.log10(P + 1e-18)
    return f, P_db

def add_awgn(x: np.ndarray, snr_db):
    if snr_db is None or np.isinf(snr_db):
        return x
    p_sig = np.mean(x**2)
    snr_lin = 10.0 ** (float(snr_db) / 10.0)
    p_noise = p_sig / snr_lin if snr_lin > 0 else 0.0
    n = np.random.normal(0.0, np.sqrt(p_noise), size=x.shape)
    return x + n

def _to_float_or_inf(s):
    if s is None:
        return None
    v = str(s).strip().lower()
    if v in ('inf','infinity'):
        return np.inf
    return float(s)

# ---------- UI page ----------

@mod_bp.route('/')
def modulation():
    return render_template('modulation.html')

# ---------- presets ----------

PRESETS = {
    'am_broadcast':     {'type':'AM','fs':4000,'t_end':1.0,'fc':400,'fm':5,'m':0.6,'snr_db':40},
    'fm_nbfm_voice':    {'type':'FM','fs':8000,'t_end':1.0,'fc':500,'fm':5,'beta':1.5,'snr_db':25},
    'pm_demo':          {'type':'PM','fs':4000,'t_end':1.0,'fc':300,'fm':5,'m':0.8,'snr_db':35},
    'pwm_slow':         {'type':'PWM','fs':2000,'t_end':1.0,'prf':20,'fm':2,'snr_db':np.inf},
    'ppm_demo':         {'type':'PPM','fs':2000,'t_end':1.0,'prf':40,'fm':3,'snr_db':np.inf},
    'pcm_16lvl':        {'type':'PCM','fs':2000,'t_end':1.0,'prf':40,'fm':4,'levels':16,'snr_db':np.inf},
}

@mod_bp.route('/api/presets')
def presets_api():
    return jsonify(list(PRESETS.keys()))

# ---------- modulation ----------

@mod_bp.route('/api/modulate')
def modulate_api():
    # collect params
    params = {k: request.args.get(k) for k in request.args.keys()}
    kind  = (params.get('type') or 'AM').upper()
    fs    = float(params.get('fs', 1000))
    t_end = float(params.get('t_end', 1.0))
    snr_db = _to_float_or_inf(params.get('snr_db', None))

    # apply preset
    preset = params.get('preset')
    if preset in PRESETS:
        for k, v in PRESETS[preset].items():
            params[k] = v
        kind  = PRESETS[preset].get('type', kind).upper()
        fs    = float(PRESETS[preset].get('fs', fs))
        t_end = float(PRESETS[preset].get('t_end', t_end))
        snr_db = _to_float_or_inf(PRESETS[preset].get('snr_db', snr_db))

    t = make_time(t_end=t_end, fs=fs)

    # placeholders
    message   = np.zeros_like(t)
    carrier   = np.zeros_like(t)
    modulated = np.zeros_like(t)
    info = {'type': kind, 'fs': fs, 'duration_s': t_end}

    # —— Analog —— #
    if kind == 'AM':
        fc = float(params.get('fc', 100))
        fm = float(params.get('fm', 5))
        m  = float(params.get('m',  0.5)); m = clamp(m, 0.0, 1.2)
        message   = np.sin(2*np.pi*fm*t)
        carrier   = np.cos(2*np.pi*fc*t)
        modulated = (1.0 + m*message) * carrier
        info.update({'fc':fc,'fm':fm,'m':m,'overmod': m>1.0})

    elif kind == 'FM':
        fc   = float(params.get('fc',   100))
        fm   = float(params.get('fm',   5))
        beta = float(params.get('beta', 5)); beta = max(0.0, beta)
        message    = np.sin(2*np.pi*fm*t)
        inst_phase = 2*np.pi*fc*t + beta*message
        modulated  = np.cos(inst_phase)
        carson_bw  = 2.0 * (beta*fm + fm)   
        info.update({'fc':fc,'fm':fm,'beta':beta,'carson_bw_hz':carson_bw})

    elif kind == 'PM':
        fc   = float(params.get('fc', 100))
        fm   = float(params.get('fm', 5))
        midx = float(params.get('m', 0.5))
        message    = np.sin(2*np.pi*fm*t)
        inst_phase = 2*np.pi*fc*t + midx*message
        modulated  = np.cos(inst_phase)
        info.update({'fc':fc,'fm':fm,'phase_index':midx})

    # —— Pulse —— #
    elif kind == 'PAM':
        prf = float(params.get('prf',50))
        fm  = float(params.get('fm', 5))
        message   = np.sin(2*np.pi*fm*t)
        pulses    = (np.mod(t,1/prf) < (1/(2*prf))).astype(float)
        carrier   = pulses.copy()
        modulated = pulses * message
        info.update({'prf':prf,'fm':fm})

    elif kind == 'PWM':
        prf = float(params.get('prf',50))
        fm  = float(params.get('fm', 5))
        message = np.sin(2*np.pi*fm*t)
        duty    = (message + 1)/2
        tau     = np.mod(t,1/prf)
        pulses  = (tau < (duty/prf)).astype(float)
        carrier   = pulses.copy()
        modulated = pulses
        info.update({'prf':prf,'fm':fm})

    elif kind == 'PPM':
        prf   = float(params.get('prf',50))
        fm    = float(params.get('fm', 5))
        message = np.sin(2*np.pi*fm*t)
        shift   = (message + 1)/2 * (1/(2*prf))
        tau     = np.mod(t,1/prf)
        width   = 1/(20*prf)
        pulses  = (np.abs(tau - shift) < width).astype(float)
        carrier   = pulses.copy()
        modulated = pulses
        info.update({'prf':prf,'fm':fm})

    elif kind == 'PCM':
        prf    = float(params.get('prf', 40))
        fm     = float(params.get('fm',  5))
        levels = int(params.get('levels', 8)); levels = max(2, levels)
        message = np.sin(2*np.pi*fm*t)
        q       = np.round((message+1)/2*(levels-1))
        qnorm   = q/(levels-1)*2 - 1
        sidx    = np.floor(t*prf).astype(int)
        carrier   = np.ones_like(t)
        modulated = qnorm[np.clip(sidx,0,len(qnorm)-1)]
        info.update({'prf':prf,'fm':fm,'levels':levels})

    else:
        return jsonify(error="Unknown modulation type"), 400

    # optional noise nd spectrum
    mod_noisy = add_awgn(modulated, snr_db)
    if snr_db is not None:
        info['snr_db'] = float(snr_db) if not np.isinf(snr_db) else 'inf'
    f_mod, Pdb_mod = spectrum_db(mod_noisy, fs)

    return jsonify({
        't': t.tolist(),
        'message': message.tolist(),
        'carrier': carrier.tolist(),
        'modulated': mod_noisy.tolist(),
        'f': f_mod.tolist(),
        'P_db': Pdb_mod.tolist(),
        'info': info
    })

# ---------- demodulation ----------

@mod_bp.route('/api/demodulate')
def demodulate_api():
    params = {k: request.args.get(k) for k in request.args.keys()}
    kind  = (params.get('type') or 'AM').upper()
    fs    = float(params.get('fs', 1000))
    t_end = float(params.get('t_end', 1.0))
    t     = make_time(t_end=t_end, fs=fs)

    if kind == 'AM':
        fc = float(params.get('fc',100))
        fm = float(params.get('fm',5))
        m  = float(params.get('m',0.5))
        msg = np.sin(2*np.pi*fm*t)
        tx  = (1 + m*msg) * np.cos(2*np.pi*fc*t)
        env   = np.abs(analytic_signal(tx))
        demod = env - np.mean(env)

    elif kind == 'FM':
        fc   = float(params.get('fc',100))
        fm   = float(params.get('fm',5))
        beta = float(params.get('beta',5))
        tx   = np.cos(2*np.pi*fc*t + beta*np.sin(2*np.pi*fm*t))
        z    = analytic_signal(tx)
        phi  = np.unwrap(np.angle(z))
        dphi = np.diff(phi) * fs               # radd/s
        inst_freq = dphi / (2*np.pi)           # Hz
        raw = inst_freq - fc
        dem = raw / (beta*fm + 1e-12) if fm > 0 else raw
        demod = np.concatenate([[dem[0]], dem])

    elif kind == 'PM':
        fc = float(params.get('fc',100))
        fm = float(params.get('fm',5))
        m  = float(params.get('m',0.5))
        tx = np.cos(2*np.pi*fc*t + m*np.sin(2*np.pi*fm*t))
        z    = analytic_signal(tx)
        phi  = np.unwrap(np.angle(z))
        base = 2*np.pi*fc*t
        deph = phi - base
        demod = deph / (m + 1e-12)

    elif kind in ['PAM','PWM','PPM','PCM']:
        prf = float(params.get('prf',50))
        fm  = float(params.get('fm',5))
        msg = np.sin(2*np.pi*fm*t)

        if kind == 'PAM':
            tx    = (np.mod(t,1/prf) < 1/(2*prf)).astype(float) * msg
            demod = msg
        elif kind == 'PWM':
            duty  = (msg + 1)/2
            tau   = np.mod(t,1/prf)
            tx    = (tau < (duty/prf)).astype(float)
            demod = msg
        elif kind == 'PPM':
            shift = (msg + 1)/2 * (1/(2*prf))
            tau   = np.mod(t,1/prf)
            width = 1/(20*prf)
            tx    = (np.abs(tau - shift) < width).astype(float)
            demod = msg
        elif kind == 'PCM':
            levels = int(params.get('levels',8))
            q      = np.round((msg+1)/2*(levels-1))
            demod  = q/(levels-1)*2 - 1
            sidx   = np.floor(t*prf).astype(int)
            tx     = demod[np.clip(sidx,0,len(demod)-1)]
    else:
        return jsonify(error="Unknown demodulation type"), 400

    f_rx, Pdb_rx = spectrum_db(demod, fs)

    return jsonify({
        't': t.tolist(),
        'modulated': tx.tolist(),
        'demodulated': demod.tolist(),
        'f': f_rx.tolist(),
        'P_db': Pdb_rx.tolist()
    })

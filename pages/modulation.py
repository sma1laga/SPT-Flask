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

def _format_lin(value: float, digits: int = 3) -> str:
    if value is None:
        return "—"
    if np.isinf(value):
        return "∞"
    if value == 0:
        return "0"
    if value >= 100:
        return f"{value:.1f}"
    if value >= 10:
        return f"{value:.2f}"
    return f"{value:.{digits}f}"

def _format_db(value: float) -> str:
    if value is None:
        return "—"
    if np.isinf(value):
        return "∞ dB"
    if value <= 0:
        return "−∞ dB"
    return f"{10*np.log10(value):.2f} dB"

def _snr_out_db(snr_in_db, factor):
    if factor is None:
        return "—"
    if factor <= 0:
        return "−∞ dB"
    if snr_in_db is None or np.isinf(snr_in_db):
        return "∞ dB"
    snr_in_lin = 10.0 ** (float(snr_in_db) / 10.0)
    snr_out_lin = snr_in_lin * factor
    if snr_out_lin <= 0:
        return "−∞ dB"
    return f"{10*np.log10(snr_out_lin):.2f} dB"

def snr_summary_am(m: float, snr_db, active_mode: str):
    m = max(0.0, float(m))
    snr_db_val = None if snr_db is None else float(snr_db)

    def entry(name, snr_formula, nf_formula, eta_formula, factor, eta_value, requires_sync=False, active=False):
        nf = np.inf if factor == 0 else (1.0 / factor)
        return {
            'scheme': name,
            'snr_formula': snr_formula,
            'nf_formula': nf_formula,
            'eta_formula': eta_formula,
            'snr_factor': _format_lin(factor),
            'snr_out_db': _snr_out_db(snr_db_val, factor),
            'nf_value': _format_lin(nf),
            'nf_db': _format_db(nf),
            'eta_value': _format_lin(eta_value),
            'requires_sync': requires_sync,
            'active': active,
        }

    m_sq = m ** 2
    am_with_factor = m_sq / 2.0
    am_with_eta = m_sq / (2.0 + m_sq) if (2.0 + m_sq) > 0 else 0.0

    summary = [
        entry(
            'AM o.Tr. (DSB-SC)',
            r"\(\mathrm{SNR}_{\text{out}} = \mathrm{SNR}_{\text{in}}\)",
            r"\(NF = 1\)",
            r"\(\eta = 1\)",
            1.0,
            1.0,
            requires_sync=True,
            active=(active_mode == 'without')
        ),
        entry(
            'AM m.Tr. (with carrier)',
            r"\(\mathrm{SNR}_{\text{out}} = \frac{m^2}{2}\, \mathrm{SNR}_{\text{in}}\)",
            r"\(NF = \frac{2}{m^2}\)",
            r"\(\eta = \frac{m^2}{2 + m^2}\)",
            am_with_factor,
            am_with_eta,
            requires_sync=False,
            active=(active_mode == 'with')
        ),
        entry(
            'QAM (coherent)',
            r"\(\mathrm{SNR}_{\text{out}} = \mathrm{SNR}_{\text{in}}\)",
            r"\(NF = 1\)",
            r"\(\eta = 1\)",
            1.0,
            1.0,
            requires_sync=True,
            active=(active_mode == 'qam')
        ),
        entry(
            'EM / RM (SSB / VSB)',
            r"\(\mathrm{SNR}_{\text{out}} \approx \mathrm{SNR}_{\text{in}}\)",
            r"\(NF \approx 1\)",
            r"\(\eta \approx 1\)",
            1.0,
            1.0,
            requires_sync=True,
            active=False
        )
    ]

    return summary


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
    'am_broadcast':     {'type':'AM','fs':4000,'t_end':1.0,'fc':400,'fm':5,'m':0.6,'snr_db':40,'carrier_mode':'with'},
    'fm_nbfm_voice':    {'type':'FM','fs':8000,'t_end':1.0,'fc':500,'fm':5,'beta':1.5,'snr_db':25},
    'pm_demo':          {'type':'PM','fs':4000,'t_end':1.0,'fc':300,'fm':5,'m':0.8,'snr_db':35},
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
    carrier_mode = (params.get('carrier_mode') or 'with').lower()


    # apply preset
    preset = params.get('preset')
    if preset in PRESETS:
        for k, v in PRESETS[preset].items():
            params[k] = v
        kind  = PRESETS[preset].get('type', kind).upper()
        fs    = float(PRESETS[preset].get('fs', fs))
        t_end = float(PRESETS[preset].get('t_end', t_end))
        snr_db = _to_float_or_inf(PRESETS[preset].get('snr_db', snr_db))
        carrier_mode = (PRESETS[preset].get('carrier_mode', carrier_mode) or carrier_mode).lower()


    t = make_time(t_end=t_end, fs=fs)

    # placeholders
    message   = np.zeros_like(t)
    carrier   = np.zeros_like(t)
    modulated = np.zeros_like(t)
    i_msg = None
    q_msg = None
    info = {'type': kind, 'fs': fs, 'duration_s': t_end}
    if kind == 'AM':
        info['carrier_mode'] = carrier_mode
    # —— Analog —— #
    if kind == 'AM':
        fc = float(params.get('fc', 100))
        fm = float(params.get('fm', 5))
        m  = float(params.get('m',  0.5)); m = clamp(m, 0.0, 1.2)
        message   = np.sin(2*np.pi*fm*t)
        carrier   = np.cos(2*np.pi*fc*t)
        if carrier_mode == 'without':
            modulated = (m * message) * carrier
        else:
            modulated = (1.0 + m*message) * carrier
            info.update({'fc':fc,'fm':fm,'m':m,'overmod': m>1.0})
        info['snr_summary'] = snr_summary_am(m, snr_db, carrier_mode)

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

    elif kind == 'QAM':
        fc   = float(params.get('fc', 200))
        fm   = float(params.get('fm', 5))
        i_amp = float(params.get('i_amp', 0.7))
        q_amp = float(params.get('q_amp', 0.7))
        i_msg = i_amp * np.cos(2*np.pi*fm*t)
        q_msg = q_amp * np.sin(2*np.pi*fm*t)
        complex_bb = i_msg + 1j*q_msg
        modulated = np.real(complex_bb * np.exp(1j*2*np.pi*fc*t))
        message = i_msg
        carrier = np.zeros_like(t)
        info.update({
            'fc': fc,
            'fm': fm,
            'i_amp': i_amp,
            'q_amp': q_amp,
            'snr_summary': snr_summary_am(1.0, snr_db, 'qam')
        })

    else:
        return jsonify(error="Unknown modulation type"), 400

    # optional noise nd spectrum
    mod_noisy = add_awgn(modulated, snr_db)
    if snr_db is not None:
        info['snr_db'] = float(snr_db) if not np.isinf(snr_db) else 'inf'
    else:
        info['snr_db'] = None
    f_mod, Pdb_mod = spectrum_db(mod_noisy, fs)

    return jsonify({
        't': t.tolist(),
        'message': message.tolist(),
        'carrier': carrier.tolist(),
        'modulated': mod_noisy.tolist(),
        'message_i': i_msg.tolist() if i_msg is not None else None,
        'message_q': q_msg.tolist() if q_msg is not None else None,
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
        carrier_mode = (params.get('carrier_mode') or 'with').lower()
        msg = np.sin(2*np.pi*fm*t)
        carrier = np.cos(2*np.pi*fc*t)
        if carrier_mode == 'without':
            tx = (m * msg) * carrier
        else:
            tx  = (1 + m*msg) * carrier
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
    elif kind == 'QAM':
        fc   = float(params.get('fc', 200))
        fm   = float(params.get('fm', 5))
        i_amp = float(params.get('i_amp', 0.7))
        q_amp = float(params.get('q_amp', 0.7))
        phase_error = np.deg2rad(float(params.get('phase_error_deg', 0.0)))
        snr_db = _to_float_or_inf(params.get('snr_db'))
        i_msg = i_amp * np.cos(2*np.pi*fm*t)
        q_msg = q_amp * np.sin(2*np.pi*fm*t)
        bb = i_msg + 1j*q_msg
        tx = np.real(bb * np.exp(1j*2*np.pi*fc*t))
        tx_noisy = add_awgn(tx, snr_db)
        analytic = analytic_signal(tx_noisy)
        lo = np.exp(-1j*(2*np.pi*fc*t + phase_error))
        bb_rec = analytic * lo
        i_rec = np.real(bb_rec)
        q_rec = np.imag(bb_rec)
        demod = i_rec
        mag = np.abs(bb_rec)
        phase = np.unwrap(np.angle(bb_rec))
        mix_i = tx_noisy * np.cos(2*np.pi*fc*t + phase_error)
        mix_q = tx_noisy * np.sin(2*np.pi*fc*t + phase_error)
        response = {
            't': t.tolist(),
            'modulated': tx_noisy.tolist(),
            'baseband_i': i_rec.tolist(),
            'baseband_q': q_rec.tolist(),
            'magnitude': mag.tolist(),
            'phase_deg': np.degrees(phase).tolist(),
            'mix_i': mix_i.tolist(),
            'mix_q': mix_q.tolist(),
            'lo_cos': np.cos(2*np.pi*fc*t + phase_error).tolist(),
            'lo_sin': np.sin(2*np.pi*fc*t + phase_error).tolist(),
            'message_i': i_msg.tolist(),
            'message_q': q_msg.tolist(),
            'phase_error_deg': float(params.get('phase_error_deg', 0.0))
        }
        f_rx, Pdb_rx = spectrum_db(i_rec, fs)
        response['f'] = f_rx.tolist()
        response['P_db'] = Pdb_rx.tolist()
        return jsonify(response)

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

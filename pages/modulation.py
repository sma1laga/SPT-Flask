# modulation.py

import numpy as np
from flask import Blueprint, render_template, request, jsonify

mod_bp = Blueprint('modulation', __name__)

# ---------- helpers ----------

MAX_PLOT_POINTS = 20000


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


def _normalize_carrier_mode(mode: str) -> str:
    mapping = {
        'with': 'with',
        'with_carrier': 'with',
        'carrier': 'with',
        'dsb_sc': 'dsb_sc',
        'dsb-sc': 'dsb_sc',
        'without': 'dsb_sc',
        'zsb': 'dsb_sc',
        'ssb_regel': 'ssb_upper',
        'ssb_upper': 'ssb_upper',
        'upper': 'ssb_upper',
        'ssb_kehr': 'ssb_lower',
        'ssb_lower': 'ssb_lower',
        'lower': 'ssb_lower',
        'vsb': 'vsb',
    }
    return mapping.get((mode or 'with').lower(), 'with')


def _carrier_mode_label(mode: str) -> str:
    labels = {
        'with': 'AM with carrier (DSB-TC)',
        'dsb_sc': 'DSB-SC (suppressed carrier)',
        'ssb_upper': 'SSB (upper sideband)',
        'ssb_lower': 'SSB (lower sideband)',
        'vsb': 'VSB (vestigial sideband)',
    }
    return labels.get(mode, mode)


def _downsample_bundle(arrays, max_points=MAX_PLOT_POINTS):
    processed = []
    lengths = []
    for arr in arrays:
        if arr is None:
            processed.append(None)
            continue
        arr_np = np.asarray(arr)
        processed.append(arr_np)
        if arr_np.size:
            lengths.append(arr_np.size)
    if not lengths:
        return processed, 1
    step = max(1, int(np.ceil(max(lengths) / max_points)))
    if step <= 1:
        return processed, 1
    downsampled = []
    for arr in processed:
        if arr is None:
            downsampled.append(None)
        elif arr.size == 0:
            downsampled.append(arr)
        else:
            downsampled.append(arr[::step])
    return downsampled, step


def _to_list(arr):
    if arr is None:
        return None
    return np.asarray(arr).tolist()


def _lowpass_fft(x: np.ndarray, fs: float, cutoff_hz: float) -> np.ndarray:
    if cutoff_hz <= 0:
        return np.real(x)
    X = np.fft.fft(x)
    freqs = np.fft.fftfreq(x.size, d=1.0/fs)
    cutoff = min(cutoff_hz, fs / 2.0)
    mask = np.abs(freqs) <= cutoff
    return np.real(np.fft.ifft(X * mask))


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
            active=(active_mode == 'dsb_sc')
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
            'EM (SSB)',
            r"\(\mathrm{SNR}_{\text{out}} \approx \mathrm{SNR}_{\text{in}}\)",
            r"\(NF \approx 1\)",
            r"\(\eta \approx 1\)",
            1.0,
            1.0,
            requires_sync=True,
            active=(active_mode in ('ssb_upper', 'ssb_lower'))
        ),
        entry(
            'VSB (vestigial)',
            r"\(\mathrm{SNR}_{\text{out}} \approx \mathrm{SNR}_{\text{in}}\)",
            r"\(NF \approx 1\)",
            r"\(\eta \approx 1\)",
            1.0,
            1.0,
            requires_sync=True,
            active=(active_mode == 'vsb')
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


    # apply preset
    preset = params.get('preset')
    if preset in PRESETS:
        for k, v in PRESETS[preset].items():
            params[k] = v
        kind  = PRESETS[preset].get('type', kind).upper()
        fs    = float(PRESETS[preset].get('fs', fs))
        t_end = float(PRESETS[preset].get('t_end', t_end))
        snr_db = _to_float_or_inf(PRESETS[preset].get('snr_db', snr_db))
    carrier_mode = _normalize_carrier_mode(params.get('carrier_mode'))


    t = make_time(t_end=t_end, fs=fs)

    # placeholders
    message   = np.zeros_like(t)
    carrier   = np.zeros_like(t)
    modulated = np.zeros_like(t)
    i_msg = None
    q_msg = None
    info = {'type': kind, 'fs': fs, 'duration_s': t_end}
    # —— Analog —— #
    if kind == 'AM':
        fc = float(params.get('fc', 100))
        fm = float(params.get('fm', 5))
        m  = float(params.get('m',  0.5)); m = clamp(m, 0.0, 1.2)
        message   = np.sin(2*np.pi*fm*t)
        carrier   = np.cos(2*np.pi*fc*t)
        sin_carrier = np.sin(2*np.pi*fc*t)
        baseband = m * message
        if carrier_mode in ('ssb_upper', 'ssb_lower', 'vsb'):
            hilbert_msg = np.imag(analytic_signal(baseband))
        else:
            hilbert_msg = None

        if carrier_mode == 'dsb_sc':
            modulated = baseband * carrier
        elif carrier_mode == 'ssb_upper':

            modulated = baseband * carrier - hilbert_msg * sin_carrier
            info['ssb_sideband'] = 'upper'
            info['badge'] = 'BW halves, SNR_out≈SNR_in'
        elif carrier_mode == 'ssb_lower':
            modulated = baseband * carrier + hilbert_msg * sin_carrier
            info['ssb_sideband'] = 'lower'
            info['badge'] = 'BW halves, SNR_out≈SNR_in'
        elif carrier_mode == 'vsb':
            dsb = baseband * carrier
            ssb_upper = baseband * carrier - hilbert_msg * sin_carrier
            modulated = 0.5 * dsb + 0.5 * ssb_upper
            info['badge'] = 'Vestigial: ≈¾ BW'
        else:
            modulated = (1.0 + m*message) * carrier
            info['overmod'] = m > 1.0

        info.update({
            'fc': fc,
            'fm': fm,
            'm': m,
            'carrier_mode': carrier_mode,
            'carrier_mode_label': _carrier_mode_label(carrier_mode)
        })
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


    plot_arrays, plot_step = _downsample_bundle([t, message, carrier, mod_noisy, i_msg, q_msg])
    t_plot, msg_plot, carrier_plot, mod_plot, i_plot, q_plot = plot_arrays
    spec_arrays, spec_step = _downsample_bundle([f_mod, Pdb_mod])
    f_plot, Pdb_plot = spec_arrays

    info['samples'] = int(t.size)
    info['plot_decimation'] = int(plot_step)
    info['spectrum_decimation'] = int(spec_step)

    return jsonify({
        't': _to_list(t_plot),
        'message': _to_list(msg_plot),
        'carrier': _to_list(carrier_plot),
        'modulated': _to_list(mod_plot),
        'message_i': _to_list(i_plot),
        'message_q': _to_list(q_plot),
        'f': _to_list(f_plot),
        'P_db': _to_list(Pdb_plot),
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
        carrier_mode = _normalize_carrier_mode(params.get('carrier_mode'))
        detector = (params.get('detector') or 'envelope').lower()
        msg = np.sin(2*np.pi*fm*t)
        carrier = np.cos(2*np.pi*fc*t)
        sin_car = np.sin(2*np.pi*fc*t)
        baseband = m * msg
        hilbert_msg = None
        if carrier_mode in ('ssb_upper', 'ssb_lower', 'vsb'):
            hilbert_msg = np.imag(analytic_signal(baseband))

        if carrier_mode == 'dsb_sc':
            tx = baseband * carrier
        elif carrier_mode == 'ssb_upper':
            tx = baseband * carrier - hilbert_msg * sin_car
        elif carrier_mode == 'ssb_lower':

            tx = baseband * carrier + hilbert_msg * sin_car
        elif carrier_mode == 'vsb':
            dsb = baseband * carrier
            ssb_upper = baseband * carrier - hilbert_msg * sin_car
            tx = 0.5 * dsb + 0.5 * ssb_upper
        else:
            tx  = (1 + baseband) * carrier

        analytic = analytic_signal(tx)
        if carrier_mode != 'with':
            detector = 'sync'

        if carrier_mode == 'with' and detector == 'envelope':
            env = np.abs(analytic)
            demod = env - np.mean(env)
        else:
            bb = analytic * np.exp(-1j*2*np.pi*fc*t)
            bb_real = np.real(bb)
            if carrier_mode == 'with':
                demod = bb_real - np.mean(bb_real)
            elif carrier_mode == 'vsb':
                cutoff = max(fm * 1.5, fm + 2.0)
                demod = _lowpass_fft(bb_real, fs, cutoff)
            else:
                demod = bb_real

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
        phase_deg = np.degrees(phase)
        lo_cos = np.cos(2*np.pi*fc*t + phase_error)
        lo_sin = np.sin(2*np.pi*fc*t + phase_error)
        f_rx, Pdb_rx = spectrum_db(i_rec, fs)

        plot_arrays, plot_step = _downsample_bundle([
            t, tx_noisy, i_rec, q_rec, mag, phase_deg,
            mix_i, mix_q, lo_cos, lo_sin, i_msg, q_msg
        ])
        (
            t_plot, tx_plot, i_plot, q_plot, mag_plot, phase_plot,
            mix_i_plot, mix_q_plot, lo_cos_plot, lo_sin_plot,
            i_msg_plot, q_msg_plot
        ) = plot_arrays
        spec_arrays, spec_step = _downsample_bundle([f_rx, Pdb_rx])
        f_plot, Pdb_plot = spec_arrays

        return jsonify({
            't': _to_list(t_plot),
            'modulated': _to_list(tx_plot),
            'baseband_i': _to_list(i_plot),
            'baseband_q': _to_list(q_plot),
            'magnitude': _to_list(mag_plot),
            'phase_deg': _to_list(phase_plot),
            'mix_i': _to_list(mix_i_plot),
            'mix_q': _to_list(mix_q_plot),
            'lo_cos': _to_list(lo_cos_plot),
            'lo_sin': _to_list(lo_sin_plot),
            'message_i': _to_list(i_msg_plot),
            'message_q': _to_list(q_msg_plot),
            'phase_error_deg': float(params.get('phase_error_deg', 0.0)),
            'f': _to_list(f_plot),
            'P_db': _to_list(Pdb_plot),
            'plot_decimation': int(plot_step),
            'spectrum_decimation': int(spec_step)
        })

    else:
        return jsonify(error="Unknown demodulation type"), 400

    f_rx, Pdb_rx = spectrum_db(demod, fs)
    plot_arrays, plot_step = _downsample_bundle([t, tx, demod])
    t_plot, tx_plot, demod_plot = plot_arrays
    spec_arrays, spec_step = _downsample_bundle([f_rx, Pdb_rx])
    f_plot, Pdb_plot = spec_arrays


    return jsonify({
        't': _to_list(t_plot),
        'modulated': _to_list(tx_plot),
        'demodulated': _to_list(demod_plot),
        'f': _to_list(f_plot),
        'P_db': _to_list(Pdb_plot),
        'plot_decimation': int(plot_step),
        'spectrum_decimation': int(spec_step)
    })

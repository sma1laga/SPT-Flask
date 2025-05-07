import numpy as np
from flask import Blueprint, render_template, request, jsonify

mod_bp = Blueprint('modulation', __name__)

def make_time(t_end=1.0, fs=1000):
    return np.linspace(0, t_end, int(fs*t_end), endpoint=False)

@mod_bp.route('/')
def modulation():
    return render_template('modulation.html')

@mod_bp.route('/api/modulate')
def modulate_api():
    kind = request.args.get('type', 'AM')
    t = make_time()

    # placeholders
    message   = np.zeros_like(t)
    carrier   = np.zeros_like(t)
    modulated = np.zeros_like(t)

    # —— Analog Modulations —— #
    if kind == 'AM':
        fc = float(request.args.get('fc',100))
        fm = float(request.args.get('fm',5))
        m  = float(request.args.get('m',0.5))
        message   = np.sin(2*np.pi*fm*t)
        carrier   = np.cos(2*np.pi*fc*t)
        modulated = (1 + m*message) * carrier

    elif kind == 'FM':
        fc   = float(request.args.get('fc',100))
        fm   = float(request.args.get('fm',5))
        beta = float(request.args.get('beta',5))
        message   = np.sin(2*np.pi*fm*t)
        inst_phase = 2*np.pi*fc*t + beta*message
        modulated  = np.cos(inst_phase)

    elif kind == 'PM':
        fc   = float(request.args.get('fc',100))
        fm   = float(request.args.get('fm',5))
        m    = float(request.args.get('m',0.5))
        message   = np.sin(2*np.pi*fm*t)
        inst_phase = 2*np.pi*fc*t + m*message
        modulated  = np.cos(inst_phase)

    # —— Pulse Modulations —— #
    elif kind == 'PAM':
        prf = float(request.args.get('prf',50))
        fm  = float(request.args.get('fm',5))
        message   = np.sin(2*np.pi*fm*t)
        pulses    = (np.mod(t,1/prf) < (1/(2*prf))).astype(float)
        carrier   = pulses.copy()
        modulated = pulses * message

    elif kind == 'PWM':
        prf = float(request.args.get('prf',50))
        fm  = float(request.args.get('fm',5))
        message = np.sin(2*np.pi*fm*t)
        duty    = (message + 1)/2
        tau     = np.mod(t,1/prf)
        pulses  = (tau < (duty/prf)).astype(float)
        carrier   = pulses.copy()
        modulated = pulses

    elif kind == 'PPM':
        prf   = float(request.args.get('prf',50))
        fm    = float(request.args.get('fm',5))
        message = np.sin(2*np.pi*fm*t)
        shift   = (message + 1)/2 * (1/(2*prf))
        tau     = np.mod(t,1/prf)
        width   = 1/(20*prf)
        pulses  = (np.abs(tau - shift) < width).astype(float)
        carrier   = pulses.copy()
        modulated = pulses

    elif kind == 'PCM':
        prf    = float(request.args.get('prf',50))
        fm     = float(request.args.get('fm',5))
        levels = int(request.args.get('levels',8))
        message = np.sin(2*np.pi*fm*t)
        q       = np.round((message+1)/2*(levels-1))
        qnorm   = q/(levels-1)*2 - 1
        sidx    = np.floor(t*prf).astype(int)
        carrier   = np.ones_like(t)
        modulated = qnorm[np.clip(sidx,0,len(qnorm)-1)]

    else:
        return jsonify(error="Unknown modulation type"), 400

    return jsonify({
        't': t.tolist(),
        'message': message.tolist(),
        'carrier': carrier.tolist(),
        'modulated': modulated.tolist()
    })


@mod_bp.route('/api/demodulate')
def demodulate_api():
    kind = request.args.get('type', 'AM')
    t = make_time()
    demod = np.zeros_like(t)
    tx    = np.zeros_like(t)

    # —— Analog Demod —— #
    if kind == 'AM':
        fc = float(request.args.get('fc',100))
        fm = float(request.args.get('fm',5))
        m  = float(request.args.get('m',0.5))
        message = np.sin(2*np.pi*fm*t)
        tx      = (1 + m*message) * np.cos(2*np.pi*fc*t)
        env     = np.abs(tx)
        demod   = env - np.mean(env)

    elif kind == 'FM':
        fc   = float(request.args.get('fc',100))
        fm   = float(request.args.get('fm',5))
        beta = float(request.args.get('beta',5))
        tx    = np.cos(2*np.pi*fc*t + beta*np.sin(2*np.pi*fm*t))
        # crude instantaneous-frequency via diff of angle
        inst = np.unwrap(np.angle(np.fft.ifft(np.fft.fft(tx)*1j)))
        dφ   = np.diff(inst)
        demod = np.concatenate([[dφ[0]], dφ])

    elif kind == 'PM':
        fc = float(request.args.get('fc',100))
        fm = float(request.args.get('fm',5))
        m  = float(request.args.get('m',0.5))
        tx = np.cos(2*np.pi*fc*t + m*np.sin(2*np.pi*fm*t))
        deph = np.unwrap(np.angle(tx)) - 2*np.pi*fc*t
        demod = deph

    # —— Pulse Demod —— #
    elif kind in ['PAM','PWM','PPM','PCM']:
        prf = float(request.args.get('prf',50))
        fm  = float(request.args.get('fm',5))
        message = np.sin(2*np.pi*fm*t)

        if kind == 'PAM':
            tx    = (np.mod(t,1/prf) < 1/(2*prf)).astype(float) * message
            demod = message

        elif kind == 'PWM':
            duty  = (message + 1)/2
            tau   = np.mod(t,1/prf)
            tx    = (tau < (duty/prf)).astype(float)
            demod = message

        elif kind == 'PPM':
            shift = (message + 1)/2 * (1/(2*prf))
            tau   = np.mod(t,1/prf)
            width = 1/(20*prf)
            tx    = (np.abs(tau - shift) < width).astype(float)
            demod = message

        elif kind == 'PCM':
            levels = int(request.args.get('levels',8))
            message = np.sin(2*np.pi*fm*t)
            q       = np.round((message+1)/2*(levels-1))
            demod   = q/(levels-1)*2 - 1
            sidx    = np.floor(t*prf).astype(int)
            tx      = demod[np.clip(sidx,0,len(demod)-1)]

    else:
        return jsonify(error="Unknown demodulation type"), 400

    return jsonify({
        't': t.tolist(),
        'modulated': tx.tolist(),
        'demodulated': demod.tolist()
    })

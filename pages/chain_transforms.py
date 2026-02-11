import numpy as np
from utils.eval_helpers import safe_eval

# -----------------------------------------------------------
def rect(w,width=1): return np.where(np.abs(w)<=width/2,1.0,0.0)
def tri (w,width=1): return np.where(np.abs(w)<=width/2,1-np.abs(w)/(width/2),0.0)
# -----------------------------------------------------------
def _freq_shift(arr,k):                          # shift array by k bins
    out=np.zeros_like(arr,dtype=complex)
    if   k>0: out[k:] =arr[:-k]
    elif k<0: out[:k]=arr[-k:]
    else:     out=arr
    return out
# -----------------------------------------------------------
def apply_addition(s,param,w):
    try: v=float(param); return s+v
    except: return s
def apply_subtraction(s,param,w):
    try: v=float(param); return s-v
    except: return s
# -----------------------------------------------------------
def apply_multiplication(signal, param, w):
    """
    Extended multiplier — now supports
      * constant:K                  (K·X)
      * imaginary[:K]               (j·K·X)
      * linear:A                    (A·ω·X)
      * sin[:J],A,ω0                if J=='j'  ⇒  j·A·sin(ω0 t)
      * cos[:J],A,ω0                if J=='j'  ⇒  j·A·cos(ω0 t)
      * exponential:K,sign,ω0       K·e^{±jω0t}  (sign = + | -)
      * sampling:T                  comb factor (1/T amplitude)
      * raw python expression       (fallback)
    """

    if not param:                    # nothing entered
        return signal

    p = param.strip().lower()

    # --- sampling comb  ------------------------------------------------
    if p.startswith("sampling:"):
        T = float(p.split(":")[1])
        tol = T / 10
        mask = np.where(np.abs(w - np.round(w/T)*T) < tol, 1.0/T, 0.0)
        return signal * mask

    # helper for spectral shift
    dw = w[1]-w[0]
    shift = lambda sig, ω0: _freq_shift(sig, int(round(ω0/dw))*-1)

    # --- constant ------------------------------------------------------
    if p.startswith("constant:"):
        K = float(p.split(":")[1])
        return K * signal

    # --- imaginary (+ optional factor) ---------------------------------
    if p.startswith("imaginary"):
        K = 1.0
        if ":" in p: K = float(p.split(":")[1])
        return 1j * K * signal

    # --- linear --------------------------------------------------------
    if p.startswith("linear:"):
        A = float(p.split(":")[1])
        return A * w * signal

    # --- sin / cos -----------------------------------------------------
    if p.startswith(("sin:","cos:")):
        is_sin = p.startswith("sin:")
        tokens = p.split(":")[1].split(",")
        jflag = False
        if tokens[0] == "j":
            jflag = True
            tokens = tokens[1:]
        A   = float(tokens[0]) if tokens and tokens[0] else 1.0
        ω0  = float(tokens[1]) if len(tokens)>1 else 1.0
        base = (A/(2j) if is_sin else A/2.0)
        if is_sin:
            y = base * ( shift(signal, ω0) - shift(signal,-ω0) )
        else:
            y = base * ( shift(signal, ω0) + shift(signal,-ω0) )
        return 1j*y if jflag else y

    # --- exponential ---------------------------------------------------
    if p.startswith("exponential:"):
        items = p.split(":")[1].split(",")
        if len(items) == 3:
            K, sign, ω0 = items
            K = float(K)
        else:
            K = 1.0
            sign, ω0 = items
        ω0  = float(ω0)
        sign = +1 if sign.strip().startswith("+") else -1
        return K * shift(signal, sign*ω0)

    # --- fallback expression ------------------------------------------
    safe = {"w":w,"sin":np.sin,"cos":np.cos,"exp":np.exp,
            "rect":rect,"tri":tri,"j":1j}
    factor = safe_eval(param, safe)
    return signal * factor

# -----------------------------------------------------------
def apply_derivative(signal,param,w): return 1j*w*signal
def apply_hilbert   (signal,param,w): return -1j*np.sign(w)*signal
def apply_real(sig,param,w): return np.real(sig)
def apply_imag(sig,param,w):
    """Return the imaginary component as a real-valued signal."""
    return np.imag(sig)# -----------------------------------------------------------
def apply_filter(signal,param,w):
    if not param: param="lowpass:1"
    mode,rest=param.lower().split(":")
    if mode=="lowpass":
        c=float(rest); mask=np.abs(w)<=c
    elif mode=="highpass":
        c=float(rest); mask=np.abs(w)>=c
    elif mode=="bandpass":
        lo,hi=[float(x) for x in rest.split(",")]; mask=(np.abs(w)>=lo)&(np.abs(w)<=hi)
    else: mask=np.ones_like(w)
    return signal*mask
# -----------------------------------------------------------
def apply_generic(signal,param,w):
    if not param: return signal
    safe={"w":w,"x":signal,"rect":rect,"tri":tri}
    return safe_eval(param,safe)
def no_op(signal,param,w): return signal


def apply_integrator(sig, param, w):
    eps = 1e-12
    return sig / (1j*w + eps)

def apply_power    (sig, p, w):  return np.abs(sig)**2
def apply_conj     (sig, p, w):  return np.conj(sig)

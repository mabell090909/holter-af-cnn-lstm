"""
Clinical Post-Processing Heuristics: Regularity Guard & Hybrid Fallback
======================================================================
Official implementation of the clinical post-processing guards for 
single-lead 10-second ECG Atrial Fibrillation detection.

These domain-knowledge rules operate as a safety layer on top of the raw
CNN-LSTM classifier probability output:

1. Clinical Regularity Guard (Tachycardia Regularity Check):
   - Overrides False Positive AF detections when the underlying ventricular rhythm 
     is excessively regular (CV_RR < 0.08 / 8%), characteristic of Sinus Tachycardia
     or regular atrial flutter rather than irregularly irregular AF.
     
2. Hybrid Fallback (Bradycardic AF Guard):
   - Mitigates False Negative AF omissions during slow ventricular response (bradycardia,
     3-8 peaks per 10s, HR 18-48 BPM) when marked R-R irregularity is present (CV_RR > 0.15 / 15%).

References:
- MethodsX (Elsevier) 2026: "Continuous Atrial Fibrillation Screening via Lightweight 
  CNN-LSTM Architecture with Dual-Stage Preprocessing and Patient-Leakage-Free Validation"
"""

import numpy as np
from scipy.signal import find_peaks


def clinical_regularity_guard(segment: np.ndarray, is_af: bool, raw_score: float, fs: int = 128):
    """
    Evaluates tachycardia regularity. If the raw AI flags AF but R-R intervals
    are strictly regular (CV_RR < 8% across >= 4 beats), override to Sinus Rhythm.
    
    Parameters
    ----------
    segment : np.ndarray
        1D normalized ECG signal array of 10-second duration (1280 samples at 128 Hz).
    is_af : bool
        Initial boolean prediction from CNN-LSTM (raw_score >= operating_threshold).
    raw_score : float
        Output sigmoid probability from the CNN-LSTM model.
    fs : int, optional
        Sampling frequency in Hz (default: 128).
        
    Returns
    -------
    is_af : bool
        Updated AF classification.
    score : float
        Updated calibrated confidence score (set to 0.10 if overridden).
    overridden : bool
        True if the Regularity Guard triggered an override.
    """
    if not is_af:
        return is_af, float(raw_score), False

    th_peak = np.max(segment) * 0.4
    if th_peak > 0.1:
        # Refractory distance ~234 ms (30 samples at 128 Hz)
        min_distance = max(1, int(0.234 * fs))
        peaks, _ = find_peaks(segment, height=th_peak, distance=min_distance)
        if len(peaks) >= 4:
            rr_intervals = np.diff(peaks)
            mean_rr = np.mean(rr_intervals)
            cv = np.std(rr_intervals) / (mean_rr + 1e-8)
            if cv < 0.08:
                # Regular rhythm with CV < 8% -> Sinus Tachycardia, not AF
                return False, 0.10, True

    return is_af, float(raw_score), False


def hybrid_fallback(segment: np.ndarray, is_af: bool, raw_score: float, fs: int = 128):
    """
    Evaluates slow ventricular response (bradycardia). If raw AI output is negative,
    but segment exhibits 3 to 8 beats (18-48 BPM) with high R-R irregularity (CV_RR > 15%),
    override to AFib.
    
    Parameters
    ----------
    segment : np.ndarray
        1D normalized ECG signal array of 10-second duration (1280 samples at 128 Hz).
    is_af : bool
        Initial boolean prediction from CNN-LSTM.
    raw_score : float
        Output sigmoid probability from the CNN-LSTM model.
    fs : int, optional
        Sampling frequency in Hz (default: 128).
        
    Returns
    -------
    is_af : bool
        Updated AF classification.
    score : float
        Updated confidence score (calibrated to min(0.85 + CV_RR, 0.99) if overridden).
    overridden : bool
        True if Hybrid Fallback triggered an override.
    """
    if is_af:
        return is_af, float(raw_score), False

    th_brady = np.max(segment) * 0.5
    if th_brady > 0.1:
        # Refractory distance ~469 ms (60 samples at 128 Hz)
        min_distance = max(1, int(0.469 * fs))
        peaks, _ = find_peaks(segment, height=th_brady, distance=min_distance)
        if 3 <= len(peaks) <= 8:
            rr_intervals = np.diff(peaks)
            mean_rr = np.mean(rr_intervals)
            cv = np.std(rr_intervals) / (mean_rr + 1e-8)
            if cv > 0.15:
                # Marked irregularity in bradycardic presentation -> Slow Ventricular Response AF
                final_score = min(0.85 + float(cv), 0.99)
                return True, float(final_score), True

    return is_af, float(raw_score), False


def apply_clinical_heuristics(segment: np.ndarray, raw_score: float, threshold: float = 0.5763, fs: int = 128):
    """
    Applies the full two-stage heuristic safety layer to a single 10-second window.
    
    Parameters
    ----------
    segment : np.ndarray
        1D ECG signal window (1280 samples at 128 Hz).
    raw_score : float
        CNN-LSTM output probability.
    threshold : float
        Operating threshold T* (calibrated value: 0.5763).
    fs : int
        Sampling frequency in Hz (default: 128).
        
    Returns
    -------
    final_pred : int
        1 if AF, 0 if non-AF.
    final_score : float
        Calibrated output confidence.
    override_reason : str
        'REGULARITY_GUARD', 'HYBRID_FALLBACK', or 'NONE'.
    """
    is_af = bool(raw_score >= threshold)
    override_reason = "NONE"
    final_score = float(raw_score)

    # 1. Regularity Guard
    is_af, final_score, overridden_reg = clinical_regularity_guard(segment, is_af, final_score, fs=fs)
    if overridden_reg:
        override_reason = "REGULARITY_GUARD"
        return int(is_af), final_score, override_reason

    # 2. Hybrid Fallback
    is_af, final_score, overridden_fb = hybrid_fallback(segment, is_af, final_score, fs=fs)
    if overridden_fb:
        override_reason = "HYBRID_FALLBACK"
        return int(is_af), final_score, override_reason

    return int(is_af), final_score, override_reason

// Pure DSP helpers for the RAFNet analyzer views (waveform / spectrogram /
// pitch track). DOM-free on purpose so vitest can exercise them in Node.
// Analysis convention: mono audio at 16 kHz; spectrogram covers 0-8 kHz.

export const ANALYSIS_RATE = 16000;
export const FFT_SIZE = 1024; // 64 ms at 16 kHz
export const SPEC_BINS = FFT_SIZE / 2 + 1; // 0 .. Nyquist (8 kHz)
export const DB_FLOOR = -80;
export const DB_CEIL = 0;

function isPowerOfTwo(n: number): boolean {
  return n > 0 && (n & (n - 1)) === 0;
}

export function hannWindow(n: number): Float32Array {
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    w[i] = 0.5 * (1 - Math.cos((2 * Math.PI * i) / (n - 1)));
  }
  return w;
}

/** Radix-2 FFT magnitudes for bins 0..N/2 of a real frame. N must be a power of two. */
export function fftMagnitude(frame: Float32Array): Float32Array {
  const n = frame.length;
  if (!isPowerOfTwo(n)) throw new Error("FFT length must be a power of two");
  const re = new Float64Array(n);
  const im = new Float64Array(n);
  for (let i = 0; i < n; i++) re[i] = frame[i];
  // Bit-reversal permutation.
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      const t = re[i];
      re[i] = re[j];
      re[j] = t;
    }
  }
  // Cooley-Tukey butterflies.
  for (let len = 2; len <= n; len <<= 1) {
    const ang = (-2 * Math.PI) / len;
    const wRe = Math.cos(ang);
    const wIm = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cRe = 1;
      let cIm = 0;
      for (let k = 0; k < len / 2; k++) {
        const uRe = re[i + k];
        const uIm = im[i + k];
        const vRe = re[i + k + len / 2] * cRe - im[i + k + len / 2] * cIm;
        const vIm = re[i + k + len / 2] * cIm + im[i + k + len / 2] * cRe;
        re[i + k] = uRe + vRe;
        im[i + k] = uIm + vIm;
        re[i + k + len / 2] = uRe - vRe;
        im[i + k + len / 2] = uIm - vIm;
        const nRe = cRe * wRe - cIm * wIm;
        cIm = cRe * wIm + cIm * wRe;
        cRe = nRe;
      }
    }
  }
  const out = new Float32Array(n / 2 + 1);
  for (let k = 0; k <= n / 2; k++) {
    out[k] = Math.sqrt(re[k] * re[k] + im[k] * im[k]) / n;
  }
  return out;
}

/** Peak bin frequency in Hz with parabolic interpolation. */
export function spectrumPeakHz(mag: Float32Array, sampleRate: number): { freq: number; mag: number } {
  let best = 0;
  for (let k = 1; k < mag.length; k++) {
    if (mag[k] > mag[best]) best = k;
  }
  const binHz = sampleRate / ((mag.length - 1) * 2);
  let peak = best;
  if (best > 0 && best < mag.length - 1) {
    const a = mag[best - 1];
    const b = mag[best];
    const c = mag[best + 1];
    const denom = a - 2 * b + c;
    if (denom !== 0) peak = best + (0.5 * (a - c)) / denom;
  }
  return { freq: peak * binHz, mag: mag[best] };
}

/** Log-magnitude column normalized to 0..1 on a fixed dB scale (honest fixed floor/ceiling). */
export function logMagnitudeColumn(mag: Float32Array): Float32Array {
  const out = new Float32Array(mag.length);
  for (let k = 0; k < mag.length; k++) {
    const db = 20 * Math.log10(mag[k] + 1e-12);
    out[k] = Math.max(0, Math.min(1, (db - DB_FLOOR) / (DB_CEIL - DB_FLOOR)));
  }
  return out;
}

/** One windowed STFT column: Hann window -> FFT -> log-magnitude 0..1. */
export function stftColumn(frame: Float32Array, window: Float32Array): Float32Array {
  if (frame.length !== window.length) throw new Error("Frame and window lengths must match");
  const w = new Float32Array(frame.length);
  for (let i = 0; i < frame.length; i++) w[i] = frame[i] * window[i];
  return logMagnitudeColumn(fftMagnitude(w));
}

const COLOR_STOPS: { t: number; rgb: [number, number, number] }[] = [
  { t: 0.0, rgb: [6, 11, 22] }, // navy-black
  { t: 0.35, rgb: [11, 59, 77] }, // deep teal
  { t: 0.55, rgb: [34, 211, 238] }, // cyan
  { t: 0.78, rgb: [251, 191, 36] }, // amber
  { t: 1.0, rgb: [248, 113, 113] }, // red
];

/** Theme colormap for spectrogram energy 0..1. */
export function themeColor(t: number): string {
  const x = Math.max(0, Math.min(1, t));
  for (let s = 0; s < COLOR_STOPS.length - 1; s++) {
    const a = COLOR_STOPS[s];
    const b = COLOR_STOPS[s + 1];
    if (x >= a.t && x <= b.t) {
      const f = (x - a.t) / (b.t - a.t || 1);
      const r = Math.round(a.rgb[0] + (b.rgb[0] - a.rgb[0]) * f);
      const g = Math.round(a.rgb[1] + (b.rgb[1] - a.rgb[1]) * f);
      const bl = Math.round(a.rgb[2] + (b.rgb[2] - a.rgb[2]) * f);
      return `rgb(${r},${g},${bl})`;
    }
  }
  const last = COLOR_STOPS[COLOR_STOPS.length - 1].rgb;
  return `rgb(${last[0]},${last[1]},${last[2]})`;
}

export function rms(samples: Float32Array): number {
  let s = 0;
  for (let i = 0; i < samples.length; i++) s += samples[i] * samples[i];
  return Math.sqrt(s / Math.max(1, samples.length));
}

export interface PitchEstimate {
  f0: number | null;
  voiced: boolean;
  clarity: number; // best cumulative-mean-normalized difference (lower = more periodic)
}

/**
 * YIN fundamental-frequency estimator with voiced/unvoiced gating.
 * Returns f0 in Hz (null when unvoiced). Voice range defaults 50-800 Hz.
 */
export function yinPitch(
  samples: Float32Array,
  sampleRate: number,
  minHz = 50,
  maxHz = 800,
  threshold = 0.1,
  clarityMax = 0.35,
  minRms = 0.01,
): PitchEstimate {
  const n = samples.length;
  if (rms(samples) < minRms) return { f0: null, voiced: false, clarity: 1 };
  const minLag = Math.max(2, Math.floor(sampleRate / maxHz));
  const maxLag = Math.min(n - 2, Math.ceil(sampleRate / minHz));
  if (maxLag <= minLag) return { f0: null, voiced: false, clarity: 1 };

  const diff = new Float64Array(maxLag + 1);
  for (let tau = minLag; tau <= maxLag; tau++) {
    let s = 0;
    for (let j = 0; j < n - tau; j++) {
      const d = samples[j] - samples[j + tau];
      s += d * d;
    }
    diff[tau] = s;
  }
  // Cumulative mean normalized difference.
  const cmnd = new Float64Array(maxLag + 1);
  cmnd[minLag] = 1;
  let running = 0;
  for (let tau = minLag; tau <= maxLag; tau++) {
    running += diff[tau];
    cmnd[tau] = running === 0 ? 1 : (diff[tau] * (tau - minLag + 1)) / running;
  }
  // First dip below threshold, then walk to the local minimum.
  let tau = -1;
  for (let t = minLag; t <= maxLag; t++) {
    if (cmnd[t] < threshold) {
      tau = t;
      while (tau + 1 <= maxLag && cmnd[tau + 1] < cmnd[tau]) tau++;
      break;
    }
  }
  let clarity = 1;
  for (let t = minLag; t <= maxLag; t++) {
    if (cmnd[t] < clarity) clarity = cmnd[t];
  }
  if (tau < 0) return { f0: null, voiced: false, clarity };
  // Parabolic interpolation around the minimum.
  let refined = tau;
  if (tau > minLag && tau < maxLag) {
    const a = cmnd[tau - 1];
    const b = cmnd[tau];
    const c = cmnd[tau + 1];
    const denom = a - 2 * b + c;
    if (denom !== 0) refined = tau + (0.5 * (a - c)) / denom;
  }
  const f0 = sampleRate / refined;
  const voiced = clarity <= clarityMax && f0 >= minHz * 0.75 && f0 <= maxHz * 1.25;
  return { f0: voiced ? f0 : null, voiced, clarity };
}

/** Linear resampler between sample rates (used when the browser mic is not 16 kHz). */
export function resampleLinear(samples: Float32Array, fromRate: number, toRate: number): Float32Array {
  if (fromRate === toRate) return samples.slice();
  const ratio = fromRate / toRate;
  const outLen = Math.max(1, Math.floor(samples.length / ratio));
  const out = new Float32Array(outLen);
  for (let i = 0; i < outLen; i++) {
    const pos = i * ratio;
    const i0 = Math.floor(pos);
    const i1 = Math.min(samples.length - 1, i0 + 1);
    const f = pos - i0;
    out[i] = samples[i0] * (1 - f) + samples[i1] * f;
  }
  return out;
}

export function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 === 1 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

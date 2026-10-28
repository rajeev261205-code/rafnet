// Unit tests for the analyzer DSP (dsp.ts). A 440 Hz sine must show a
// ~440 Hz spectral peak and a ~440 Hz pitch estimate; silence is unvoiced.
import { describe, expect, it } from "vitest";
import {
  ANALYSIS_RATE,
  FFT_SIZE,
  hannWindow,
  fftMagnitude,
  spectrumPeakHz,
  stftColumn,
  themeColor,
  rms,
  yinPitch,
  resampleLinear,
  median,
} from "./dsp";

function sineWave(freq: number, sampleRate: number, seconds: number, amp = 0.5): Float32Array {
  const n = Math.floor(sampleRate * seconds);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    out[i] = amp * Math.sin((2 * Math.PI * freq * i) / sampleRate);
  }
  return out;
}

describe("spectral analysis", () => {
  it("finds the 440 Hz peak of a 440 Hz sine", () => {
    const tone = sineWave(440, ANALYSIS_RATE, 0.5);
    const frame = tone.slice(0, FFT_SIZE);
    const window = hannWindow(FFT_SIZE);
    const win = new Float32Array(FFT_SIZE);
    for (let i = 0; i < FFT_SIZE; i++) win[i] = frame[i] * window[i];
    const { freq, mag } = spectrumPeakHz(fftMagnitude(win), ANALYSIS_RATE);
    expect(mag).toBeGreaterThan(0);
    expect(freq).toBeGreaterThan(420);
    expect(freq).toBeLessThan(460);
  });

  it("stftColumn returns normalized 0..1 energy with the peak near 440 Hz", () => {
    const tone = sineWave(440, ANALYSIS_RATE, 0.5);
    const window = hannWindow(FFT_SIZE);
    const col = stftColumn(tone.slice(0, FFT_SIZE), window);
    expect(col.length).toBe(FFT_SIZE / 2 + 1);
    let best = 0;
    for (let k = 1; k < col.length; k++) {
      if (col[k] > col[best]) best = k;
    }
    const binHz = ANALYSIS_RATE / FFT_SIZE;
    expect(best * binHz).toBeGreaterThan(400);
    expect(best * binHz).toBeLessThan(480);
    for (const v of col) {
      expect(v).toBeGreaterThanOrEqual(0);
      expect(v).toBeLessThanOrEqual(1);
    }
  });

  it("themeColor maps 0..1 to css colors", () => {
    expect(themeColor(0)).toMatch(/^rgb\(/);
    expect(themeColor(1)).toMatch(/^rgb\(/);
    expect(themeColor(0)).not.toBe(themeColor(1));
  });
});

describe("pitch tracking (YIN)", () => {
  it("estimates ~440 Hz for a 440 Hz sine", () => {
    const tone = sineWave(440, ANALYSIS_RATE, 0.5);
    const est = yinPitch(tone.slice(0, 2048), ANALYSIS_RATE);
    expect(est.voiced).toBe(true);
    expect(est.f0).not.toBeNull();
    expect(est.f0 as number).toBeGreaterThan(430);
    expect(est.f0 as number).toBeLessThan(450);
  });

  it("marks silence as unvoiced", () => {
    const est = yinPitch(new Float32Array(2048), ANALYSIS_RATE);
    expect(est.voiced).toBe(false);
    expect(est.f0).toBeNull();
    expect(rms(new Float32Array(2048))).toBe(0);
  });
});

describe("helpers", () => {
  it("resampleLinear preserves a 440 Hz tone from 48 kHz to 16 kHz", () => {
    const hi = sineWave(440, 48000, 0.25);
    const lo = resampleLinear(hi, 48000, ANALYSIS_RATE);
    expect(lo.length).toBe(Math.floor(hi.length / 3));
    const window = hannWindow(FFT_SIZE);
    const frame = lo.slice(0, FFT_SIZE);
    const win = new Float32Array(FFT_SIZE);
    for (let i = 0; i < FFT_SIZE; i++) win[i] = frame[i] * window[i];
    const { freq } = spectrumPeakHz(fftMagnitude(win), ANALYSIS_RATE);
    expect(freq).toBeGreaterThan(420);
    expect(freq).toBeLessThan(460);
  });

  it("median summarizes pitch stats", () => {
    expect(median([])).toBeNull();
    expect(median([440])).toBe(440);
    expect(median([100, 200, 440])).toBe(200);
  });
});

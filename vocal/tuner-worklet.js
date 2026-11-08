// Vocal tuner DSP: YIN pitch tracking + TD-PSOLA pitch correction, scale snapping,
// octave-down layer and a look-ahead noise gate. Runs in an AudioWorklet; the core
// class is plain JS so it can also be tested offline in Node.

const SCALES = {
  minor: [0, 2, 3, 5, 7, 8, 10],
  major: [0, 2, 4, 5, 7, 9, 11],
  harmonic: [0, 2, 3, 5, 7, 8, 11],
  penta: [0, 3, 5, 7, 10],
  chromatic: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
};

class PsolaVoice {
  constructor(t) {
    this.t = t;
    this.ON = 8192;
    this.om = this.ON - 1;
    this.out = new Float32Array(this.ON);
    this.nextMark = t.Tmax;
    this.prevIn = 0;
    this.wasVoiced = false;
  }

  // Place one pitch-synchronous grain centred on output mark `nextMark`.
  place(n, ratio, voiced) {
    const t = this.t;
    const ti = this.nextMark - t.L; // matching input time (fixed latency L)
    let Tin, Tout, ci;
    if (voiced) {
      Tin = t.period;
      Tout = Tin / ratio;
      if (!this.wasVoiced) ci = ti;
      else ci = this.prevIn + Math.round((ti - this.prevIn) / Tin) * Tin;
      while (ci + Tin + 1 > n) ci -= Tin;
      if (Math.abs(ci - ti) > 2 * Tin) ci = ti; // safety
      this.wasVoiced = true;
    } else {
      Tin = Tout = t.Tu;
      ci = ti;
      this.wasVoiced = false;
    }
    this.prevIn = ci;
    const H = Math.max(2, Math.round(Tin));
    const mark = Math.round(this.nextMark);
    const g = voiced ? Math.min(2, Tout / H) : 1;
    const inBuf = t.inBuf, im = t.inMask, out = this.out, om = this.om;
    const k = Math.PI / H;
    for (let o = -H + 1; o < H; o++) {
      const w = 0.5 * (1 + Math.cos(k * o)) * g;
      const pos = ci + o;
      const i0 = Math.floor(pos);
      const fr = pos - i0;
      const a = inBuf[i0 & im], b = inBuf[(i0 + 1) & im];
      out[(mark + o) & om] += w * (a + (b - a) * fr);
    }
    this.nextMark += Tout;
  }
}

class VocalTuner {
  constructor(sr) {
    this.sr = sr;
    this.minF = 75;
    this.maxF = 900;
    this.Tmax = Math.ceil(sr / this.minF);
    this.L = Math.ceil(this.Tmax * 2.5); // processing latency in samples
    this.Tu = Math.round(sr * 0.005); // grain hop when unvoiced
    this.IN = 32768;
    this.inMask = this.IN - 1;
    this.inBuf = new Float32Array(this.IN);
    this.n = 0;

    // Pitch tracker (YIN on a low-passed, decimated copy)
    this.dec = Math.max(1, Math.round(sr / 11025));
    this.dsr = sr / this.dec;
    this.maxTau = Math.ceil(this.dsr / this.minF);
    this.minTau = Math.max(2, Math.floor(this.dsr / this.maxF));
    this.W = this.maxTau;
    this.DN = 4096;
    this.dMask = this.DN - 1;
    this.dBuf = new Float32Array(this.DN);
    this.dw = 0;
    this.decCnt = 0;
    this.frame = new Float32Array(this.W + this.maxTau + 1);
    this.dn = new Float32Array(this.maxTau + 2);
    this.hop = Math.round(sr * 0.0053);
    this.hopCnt = 0;
    // RBJ low-pass ~1 kHz
    const w0 = (2 * Math.PI * 1000) / sr, al = Math.sin(w0) / (2 * 0.707), c = Math.cos(w0);
    const a0 = 1 + al;
    this.lb0 = (1 - c) / 2 / a0; this.lb1 = (1 - c) / a0; this.lb2 = this.lb0;
    this.la1 = (-2 * c) / a0; this.la2 = (1 - al) / a0;
    this.lx1 = this.lx2 = this.ly1 = this.ly2 = 0;

    this.period = 0;
    this.voiced = false;
    this.vRun = 0;
    this.uRun = 0;
    this.octSuspect = 0;
    this.midiIn = 0;
    this.note = -1;
    this.shiftSm = 0;
    this.ratio = 1;

    // params
    this.key = 0;
    this.scaleSet = new Set(SCALES.minor);
    this.amount = 1;
    this.octave = 0.3;
    this.transpose = 0;
    this.gateDb = -52;

    this.env = 0;
    this.gateOpen = false;
    this.gateG = 0;
    this.octG = 0;
    this.aEnvA = Math.exp(-1 / (sr * 0.001));
    this.aEnvR = Math.exp(-1 / (sr * 0.08));
    this.aGateA = Math.exp(-1 / (sr * 0.002));
    this.aGateR = Math.exp(-1 / (sr * 0.15));
    this.aOct = Math.exp(-1 / (sr * 0.012));

    this.main = new PsolaVoice(this);
    this.low = new PsolaVoice(this);
    this.statusEvery = Math.round(sr * 0.05);
    this.statusCnt = 0;
    this.onStatus = null;
  }

  setParams(p) {
    if (p.key !== undefined) this.key = ((p.key % 12) + 12) % 12;
    if (p.scale !== undefined && SCALES[p.scale]) this.scaleSet = new Set(SCALES[p.scale]);
    if (p.amount !== undefined) this.amount = Math.min(1, Math.max(0, p.amount));
    if (p.octave !== undefined) this.octave = Math.max(0, p.octave);
    if (p.transpose !== undefined) this.transpose = p.transpose;
    if (p.gateDb !== undefined) this.gateDb = p.gateDb;
  }

  inScale(m) {
    return this.scaleSet.has((((m - this.key) % 12) + 12) % 12);
  }

  snap(midi) {
    const base = Math.round(midi);
    let best = base, bd = 1e9;
    for (let d = -3; d <= 3; d++) {
      const c = base + d;
      if (!this.inScale(c)) continue;
      const dist = Math.abs(c - midi);
      if (dist < bd) { bd = dist; best = c; }
    }
    if (this.note >= 0 && this.inScale(this.note) && Math.abs(midi - this.note) < bd + 0.2) return this.note;
    return best;
  }

  detect() {
    const W = this.W, mt = this.maxTau, len = W + mt;
    const f = this.frame, dm = this.dMask;
    let start = this.dw - len, e = 0;
    for (let j = 0; j < len; j++) { const v = this.dBuf[(start + j) & dm]; f[j] = v; e += v * v; }
    const rms = Math.sqrt(e / len);
    const dn = this.dn;
    dn[0] = 1;
    let running = 0;
    for (let tau = 1; tau <= mt; tau++) {
      let s = 0;
      for (let j = 0; j < W; j++) { const d = f[j] - f[j + tau]; s += d * d; }
      running += s;
      dn[tau] = running > 0 ? (s * tau) / running : 1;
    }
    let te = -1;
    for (let tau = this.minTau; tau < mt; tau++) {
      if (dn[tau] < 0.15) {
        while (tau + 1 < mt && dn[tau + 1] < dn[tau]) tau++;
        te = tau;
        break;
      }
    }
    const thrLin = Math.pow(10, (this.gateDb + 4) / 20);
    let voicedNow = te > 0 && rms > thrLin * 0.5;
    let period = 0;
    if (voicedNow) {
      const a = dn[te - 1], b = dn[te], c = dn[te + 1];
      const den = a - 2 * b + c;
      let off = den !== 0 ? (a - c) / (2 * den) : 0;
      if (off > 1) off = 1; else if (off < -1) off = -1;
      period = (te + off) * this.dec;
    }

    if (voicedNow) {
      this.uRun = 0;
      this.vRun++;
      if (this.voiced && this.period > 0) {
        const r = Math.abs(Math.log2(period / this.period));
        if (r > 0.85 && r < 1.15 && this.octSuspect < 3) { this.octSuspect++; period = this.period; }
        else this.octSuspect = 0;
      }
      if (!this.voiced && this.vRun >= 2) { this.voiced = true; this.note = -1; this.onset = true; }
      if (this.voiced) this.period = period;
      else this.period = period;
    } else {
      this.vRun = 0;
      this.uRun++;
      if (this.voiced && this.uRun >= 3) this.voiced = false;
    }

    if (this.voiced && this.period > 0) {
      const freq = this.sr / this.period;
      const midi = 69 + 12 * Math.log2(freq / 440);
      this.midiIn = midi;
      const note = this.snap(midi);
      this.note = note;
      const strength = Math.min(1, this.amount * 2);
      const target = (note - midi) * strength;
      const retuneMs = (1 - this.amount) * 160;
      const hopSec = this.hop / this.sr;
      const a = retuneMs <= 0.5 ? 1 : 1 - Math.exp(-hopSec / (retuneMs / 1000));
      if (this.onset) { this.shiftSm = target; this.onset = false; }
      else this.shiftSm += (target - this.shiftSm) * a;
      this.ratio = Math.pow(2, (this.shiftSm + this.transpose) / 12);
    } else {
      this.ratio = Math.pow(2, this.transpose / 12);
    }
  }

  process(input, output) {
    const N = output.length;
    const im = this.inMask;
    const thr = Math.pow(10, this.gateDb / 20);
    const main = this.main, low = this.low;
    const Tmax = this.Tmax;
    for (let i = 0; i < N; i++) {
      const x = input ? input[i] : 0;
      const n = this.n;
      this.inBuf[n & im] = x;

      // envelope + gate (look-ahead: env is L samples ahead of output)
      const ax = Math.abs(x);
      this.env = ax > this.env ? this.aEnvA * this.env + (1 - this.aEnvA) * ax : this.aEnvR * this.env + (1 - this.aEnvR) * ax;
      if (this.env > thr) this.gateOpen = true; else if (this.env < thr * 0.6) this.gateOpen = false;
      const gt = this.gateOpen ? 1 : 0;
      const ga = gt > this.gateG ? this.aGateA : this.aGateR;
      this.gateG = ga * this.gateG + (1 - ga) * gt;

      // low-pass + decimate for the tracker
      const y = this.lb0 * x + this.lb1 * this.lx1 + this.lb2 * this.lx2 - this.la1 * this.ly1 - this.la2 * this.ly2;
      this.lx2 = this.lx1; this.lx1 = x; this.ly2 = this.ly1; this.ly1 = y;
      if (++this.decCnt >= this.dec) { this.decCnt = 0; this.dBuf[this.dw & this.dMask] = y; this.dw++; }
      if (++this.hopCnt >= this.hop) { this.hopCnt = 0; if (this.dw > this.W + this.maxTau) this.detect(); }

      const voiced = this.voiced && this.period > 0;
      while (main.nextMark - Tmax <= n) main.place(n, this.ratio, voiced);
      while (low.nextMark - Tmax <= n) low.place(n, this.ratio * 0.5, voiced);

      const om = main.om;
      const ym = main.out[n & om]; main.out[n & om] = 0;
      const yl = low.out[n & om]; low.out[n & om] = 0;
      const oa = this.aOct;
      this.octG = oa * this.octG + (1 - oa) * (voiced ? 1 : 0);
      output[i] = (ym + yl * this.octave * this.octG) * this.gateG;
      this.n = n + 1;
    }
    if (this.onStatus && (this.statusCnt += N) >= this.statusEvery) {
      this.statusCnt = 0;
      this.onStatus({ v: this.voiced, m: this.midiIn, note: this.note, lvl: this.env });
    }
  }
}

if (typeof registerProcessor === 'function') {
  class TunerProcessor extends AudioWorkletProcessor {
    constructor() {
      super();
      this.t = new VocalTuner(sampleRate);
      this.port.onmessage = (e) => this.t.setParams(e.data);
      this.t.onStatus = (s) => this.port.postMessage(s);
      this.silence = new Float32Array(128);
    }
    process(inputs, outputs) {
      const out = outputs[0];
      if (!out || !out[0]) return true;
      const inp = inputs[0] && inputs[0][0] ? inputs[0][0] : null;
      if (!inp && this.silence.length !== out[0].length) this.silence = new Float32Array(out[0].length);
      this.t.process(inp || this.silence, out[0]);
      for (let c = 1; c < out.length; c++) out[c].set(out[0]);
      return true;
    }
  }
  registerProcessor('vocal-tuner', TunerProcessor);

  // Captures the raw mic as PCM so a take can be replayed through the chain.
  class PcmTap extends AudioWorkletProcessor {
    constructor() {
      super();
      this.on = false;
      this.buf = new Float32Array(16384);
      this.i = 0;
      this.port.onmessage = (e) => {
        if (e.data.rec) { this.on = true; this.i = 0; }
        else { this.flush(); this.on = false; this.port.postMessage({ done: true }); }
      };
    }
    flush() {
      if (this.i) this.port.postMessage({ chunk: this.buf.slice(0, this.i) });
      this.i = 0;
    }
    process(inputs) {
      const x = inputs[0] && inputs[0][0];
      if (this.on && x) {
        for (let k = 0; k < x.length; k++) {
          this.buf[this.i++] = x[k];
          if (this.i === this.buf.length) this.flush();
        }
      }
      return true;
    }
  }
  registerProcessor('pcm-tap', PcmTap);
}

if (typeof module !== 'undefined' && module.exports) module.exports = { VocalTuner, SCALES };

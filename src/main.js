import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const TAU = Math.PI * 2;
const clamp = (v, min = 0, max = 1) => Math.min(max, Math.max(min, v));
const lerp = (a, b, t) => a + (b - a) * t;
const smoothstep = (a, b, x) => {
  const t = clamp((x - a) / (b - a));
  return t * t * (3 - 2 * t);
};
const formatTime = (seconds) => {
  if (!Number.isFinite(seconds)) return '0:00';
  const s = Math.max(0, Math.floor(seconds));
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, '0')}`;
};
const nextFrame = () => new Promise((resolve) => requestAnimationFrame(resolve));

function hash32(a) {
  a |= 0;
  a = (a + 0x7ed55d16) + (a << 12);
  a = (a ^ 0xc761c23c) ^ (a >>> 19);
  a = (a + 0x165667b1) + (a << 5);
  a = (a + 0xd3a2646c) ^ (a << 9);
  a = (a + 0xfd7046c5) + (a << 3);
  a = (a ^ 0xb55a4f09) ^ (a >>> 16);
  return a >>> 0;
}

function randAt(seed, index, channel = 0) {
  return hash32(seed ^ hash32(index * 374761393 + channel * 668265263)) / 4294967296;
}

function valueNoise1D(x, seed, channel = 0) {
  const i = Math.floor(x);
  const f = x - i;
  const u = f * f * (3 - 2 * f);
  const a = randAt(seed, i, channel);
  const b = randAt(seed, i + 1, channel);
  return lerp(a, b, u);
}

function percentile(values, q) {
  if (!values.length) return 0;
  const sorted = Array.from(values).sort((a, b) => a - b);
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  return sorted[base + 1] === undefined ? sorted[base] : lerp(sorted[base], sorted[base + 1], rest);
}

function normalizeRobust(values, lowQ = 0.08, highQ = 0.94) {
  const lo = percentile(values, lowQ);
  const hi = percentile(values, highQ);
  const range = Math.max(1e-9, hi - lo);
  return Array.from(values, (v) => clamp((v - lo) / range));
}

function fftInPlace(real, imag) {
  const n = real.length;
  let j = 0;
  for (let i = 1; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) {
      [real[i], real[j]] = [real[j], real[i]];
      [imag[i], imag[j]] = [imag[j], imag[i]];
    }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const angle = -TAU / len;
    const wLenCos = Math.cos(angle);
    const wLenSin = Math.sin(angle);
    for (let i = 0; i < n; i += len) {
      let wCos = 1;
      let wSin = 0;
      for (let k = 0; k < len / 2; k++) {
        const uR = real[i + k];
        const uI = imag[i + k];
        const vR = real[i + k + len / 2] * wCos - imag[i + k + len / 2] * wSin;
        const vI = real[i + k + len / 2] * wSin + imag[i + k + len / 2] * wCos;
        real[i + k] = uR + vR;
        imag[i + k] = uI + vI;
        real[i + k + len / 2] = uR - vR;
        imag[i + k + len / 2] = uI - vI;
        const nextCos = wCos * wLenCos - wSin * wLenSin;
        wSin = wCos * wLenSin + wSin * wLenCos;
        wCos = nextCos;
      }
    }
  }
}

async function analyzeAudioBuffer(buffer, onProgress = () => {}) {
  const sampleRate = buffer.sampleRate;
  const duration = buffer.duration;
  const frameSize = 1024;
  const frameInterval = duration > 1200 ? 0.24 : duration > 480 ? 0.18 : 0.12;
  const frameCount = Math.max(2, Math.ceil(duration / frameInterval));
  const channelCount = buffer.numberOfChannels;
  const channels = Array.from({ length: channelCount }, (_, i) => buffer.getChannelData(i));
  const window = new Float32Array(frameSize);
  for (let i = 0; i < frameSize; i++) window[i] = 0.5 - 0.5 * Math.cos((TAU * i) / (frameSize - 1));

  const rawEnergy = new Float32Array(frameCount);
  const rawLow = new Float32Array(frameCount);
  const rawMid = new Float32Array(frameCount);
  const rawHigh = new Float32Array(frameCount);
  const rawCentroid = new Float32Array(frameCount);
  const rawFlux = new Float32Array(frameCount);
  const rawDominant = new Float32Array(frameCount);
  const real = new Float64Array(frameSize);
  const imag = new Float64Array(frameSize);
  const prevMag = new Float64Array(frameSize / 2);
  const currentMag = new Float64Array(frameSize / 2);
  const nyquist = sampleRate / 2;
  const binHz = sampleRate / frameSize;

  for (let frame = 0; frame < frameCount; frame++) {
    const center = Math.floor(frame * frameInterval * sampleRate);
    let sumSq = 0;
    let zc = 0;
    let previous = 0;
    for (let i = 0; i < frameSize; i++) {
      const sourceIndex = center + i - Math.floor(frameSize / 2);
      let sample = 0;
      if (sourceIndex >= 0 && sourceIndex < buffer.length) {
        for (let c = 0; c < channelCount; c++) sample += channels[c][sourceIndex] || 0;
        sample /= channelCount;
      }
      if (i > 0 && (sample >= 0) !== (previous >= 0)) zc++;
      previous = sample;
      sumSq += sample * sample;
      real[i] = sample * window[i];
      imag[i] = 0;
    }

    fftInPlace(real, imag);
    let low = 0;
    let mid = 0;
    let high = 0;
    let weighted = 0;
    let magSum = 0;
    let flux = 0;
    let dominantMag = 0;
    let dominantFreq = 0;
    for (let k = 1; k < frameSize / 2; k++) {
      const frequency = k * binHz;
      const magnitude = Math.hypot(real[k], imag[k]);
      currentMag[k] = magnitude;
      if (frequency < 180) low += magnitude;
      else if (frequency < 2200) mid += magnitude;
      else if (frequency < 11000) high += magnitude;
      if (frequency >= 35 && frequency <= 5500 && magnitude > dominantMag) {
        dominantMag = magnitude;
        dominantFreq = frequency;
      }
      const positive = magnitude - prevMag[k];
      if (positive > 0) flux += positive;
      weighted += frequency * magnitude;
      magSum += magnitude;
    }

    rawEnergy[frame] = Math.sqrt(sumSq / frameSize);
    rawLow[frame] = Math.log1p(low);
    rawMid[frame] = Math.log1p(mid);
    rawHigh[frame] = Math.log1p(high);
    rawCentroid[frame] = magSum > 0 ? clamp(weighted / magSum / nyquist) : 0;
    rawFlux[frame] = Math.log1p(flux);
    rawDominant[frame] = dominantFreq;
    prevMag.set(currentMag);

    if (frame % 24 === 0) {
      onProgress(frame / frameCount * 0.86);
      await nextFrame();
    }
  }

  const energy = normalizeRobust(rawEnergy);
  const low = normalizeRobust(rawLow);
  const mid = normalizeRobust(rawMid);
  const high = normalizeRobust(rawHigh);
  const fluxNorm = normalizeRobust(rawFlux, 0.2, 0.97);
  const centroidRaw = normalizeRobust(rawCentroid, 0.05, 0.95);
  const centroid = centroidRaw.map((v, i) => clamp(v * 0.8 + high[i] * 0.2));

  const onset = new Float32Array(frameCount);
  for (let i = 0; i < frameCount; i++) {
    const left = fluxNorm[Math.max(0, i - 1)];
    const right = fluxNorm[Math.min(frameCount - 1, i + 1)];
    const localPeak = fluxNorm[i] >= left && fluxNorm[i] >= right ? 1 : 0.55;
    const neighborhood = (
      fluxNorm[Math.max(0, i - 3)] + fluxNorm[Math.max(0, i - 2)] + fluxNorm[Math.max(0, i - 1)] +
      fluxNorm[Math.min(frameCount - 1, i + 1)] + fluxNorm[Math.min(frameCount - 1, i + 2)] + fluxNorm[Math.min(frameCount - 1, i + 3)]
    ) / 6;
    onset[i] = clamp((fluxNorm[i] - neighborhood * 0.58 - 0.08) * 2.25) * localPeak;
  }

  const targetSectionCount = duration < 12 ? 2 : clamp(Math.round(duration / 38) + 2, 3, 10);
  const radius = Math.max(3, Math.round(1.3 / frameInterval));
  const changeScores = [];
  for (let i = radius; i < frameCount - radius; i++) {
    let before = [0, 0, 0, 0, 0];
    let after = [0, 0, 0, 0, 0];
    for (let k = 1; k <= radius; k++) {
      before[0] += energy[i - k]; before[1] += low[i - k]; before[2] += mid[i - k]; before[3] += high[i - k]; before[4] += centroid[i - k];
      after[0] += energy[i + k]; after[1] += low[i + k]; after[2] += mid[i + k]; after[3] += high[i + k]; after[4] += centroid[i + k];
    }
    const inv = 1 / radius;
    let distance = 0;
    for (let j = 0; j < 5; j++) {
      const d = (after[j] - before[j]) * inv;
      distance += d * d;
    }
    const score = Math.sqrt(distance) + onset[i] * 0.28;
    changeScores.push({ i, score });
  }
  changeScores.sort((a, b) => b.score - a.score);
  const boundaries = [0];
  const minSpacing = Math.max(4.5, duration / (targetSectionCount * 2.2));
  for (const candidate of changeScores) {
    const t = candidate.i * frameInterval;
    if (t < minSpacing || duration - t < minSpacing) continue;
    if (boundaries.every((b) => Math.abs(b - t) >= minSpacing)) boundaries.push(t);
    if (boundaries.length >= targetSectionCount) break;
  }
  boundaries.push(duration);
  boundaries.sort((a, b) => a - b);

  const sections = [];
  for (let s = 0; s < boundaries.length - 1; s++) {
    const start = boundaries[s];
    const end = boundaries[s + 1];
    const a = Math.max(0, Math.floor(start / frameInterval));
    const b = Math.min(frameCount, Math.ceil(end / frameInterval));
    const averages = { energy: 0, low: 0, mid: 0, high: 0, centroid: 0 };
    const count = Math.max(1, b - a);
    for (let i = a; i < b; i++) {
      averages.energy += energy[i];
      averages.low += low[i];
      averages.mid += mid[i];
      averages.high += high[i];
      averages.centroid += centroid[i];
    }
    for (const key of Object.keys(averages)) averages[key] /= count;
    sections.push({ start, end, averages, motif: s % 4 });
  }

  onProgress(1);
  return {
    duration,
    frameInterval,
    energy: Float32Array.from(energy),
    low: Float32Array.from(low),
    mid: Float32Array.from(mid),
    high: Float32Array.from(high),
    centroid: Float32Array.from(centroid),
    onset,
    dominant: rawDominant,
    sections,
  };
}

function sampleAnalysis(analysis, time) {
  if (!analysis) return { energy: 0, low: 0, mid: 0, high: 0, centroid: 0, onset: 0, dominant: 0, sectionIndex: 0, sectionPhase: 0 };
  const maxIndex = analysis.energy.length - 1;
  const frame = clamp(time / analysis.frameInterval, 0, maxIndex);
  const i = Math.floor(frame);
  const j = Math.min(maxIndex, i + 1);
  const t = frame - i;
  let sectionIndex = analysis.sections.findIndex((s) => time >= s.start && time < s.end);
  if (sectionIndex < 0) sectionIndex = analysis.sections.length - 1;
  const section = analysis.sections[Math.max(0, sectionIndex)];
  return {
    energy: lerp(analysis.energy[i], analysis.energy[j], t),
    low: lerp(analysis.low[i], analysis.low[j], t),
    mid: lerp(analysis.mid[i], analysis.mid[j], t),
    high: lerp(analysis.high[i], analysis.high[j], t),
    centroid: lerp(analysis.centroid[i], analysis.centroid[j], t),
    onset: Math.max(analysis.onset[i], analysis.onset[j] * t),
    dominant: lerp(analysis.dominant[i], analysis.dominant[j], t),
    sectionIndex: Math.max(0, sectionIndex),
    sectionPhase: clamp((time - section.start) / Math.max(0.001, section.end - section.start)),
  };
}

const PALETTES = {
  aurora: {
    name: '夜色极光',
    dark: true,
    bg: '#060a12',
    bg2: '#11192a',
    baseHue: 0.48,
    hueRange: 0.31,
    saturation: 0.77,
    lightness: 0.61,
    accent: '#7df9ff',
  },
  ember: {
    name: '熔岩余烬',
    dark: true,
    bg: '#100809',
    bg2: '#24100f',
    baseHue: 0.96,
    hueRange: 0.17,
    saturation: 0.82,
    lightness: 0.62,
    accent: '#ff9970',
  },
  porcelain: {
    name: '青瓷墨色',
    dark: false,
    bg: '#e9ebe6',
    bg2: '#cfd8d1',
    baseHue: 0.49,
    hueRange: 0.18,
    saturation: 0.50,
    lightness: 0.38,
    accent: '#1d746f',
  },
  ultraviolet: {
    name: '紫外梦境',
    dark: true,
    bg: '#090612',
    bg2: '#201038',
    baseHue: 0.72,
    hueRange: 0.24,
    saturation: 0.78,
    lightness: 0.65,
    accent: '#c7a6ff',
  },
};


const PAINTING_STYLES = {
  trace: {
    order: 1,
    name: '声纹流线',
    label: '风格 1 · 声纹流线',
    description: '完整保留原有生成方式：低频色带、旋律主线、高频细线与瞬态叶片沿时间连续生长。它更接近声音轨迹与生成图案。',
    hint: '风格 1 · 时间从左向右，声音凝固为流线、色带与叶片',
    mappingHtml: '时间 → 横向轨迹<br>低频 → 连续色带<br>中频 → 旋律主线<br>高频 → 细线与光点<br>瞬态 → 叶片与节点<br>段落 → 色场转折',
    exportName: '风格1_声纹流线',
  },
  painterly: {
    order: 2,
    name: '声色画境',
    label: '风格 2 · 声色画境',
    description: '以绘画而不是曲线为目标：音乐逐层沉积为底色、色块、干湿笔触、刮刀印记与飞白；段落先决定构图，音符再落入不同画面区域。',
    hint: '风格 2 · 音乐逐层沉积为色场、笔触与节奏印记',
    mappingHtml: '时间 → 画面逐步完成<br>低频 → 底层色块与厚涂<br>中频 / 音高 → 主笔触与落点<br>高频 → 飞白与颜料颗粒<br>瞬态 → 刮刀与书写性重笔<br>段落 → 画面重心、留白与主题形体',
    exportName: '风格2_声色画境',
  },
};

function paletteHsl(palette, feature, progress, sectionIndex, seed, lightShift = 0) {
  const sectionOffset = (randAt(seed, sectionIndex, 91) - 0.5) * 0.14;
  const hue = (palette.baseHue + feature.centroid * palette.hueRange + progress * 0.08 + sectionOffset + 1) % 1;
  const saturation = clamp(palette.saturation + feature.high * 0.09 - feature.low * 0.06);
  const lightness = clamp(palette.lightness + (feature.energy - 0.5) * 0.16 + lightShift, 0.18, 0.84);
  return { hue, saturation, lightness };
}

class CanvasPainting {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d', { alpha: false });
    this.width = 0;
    this.height = 0;
    this.dpr = 1;
    this.prev = null;
    this.painterPrev = null;
    this.painterGroundPrev = null;
    this.lastSection = -1;
    this.palette = PALETTES.aurora;
    this.seed = 1;
    this.complexity = 1;
    this.duration = 1;
    this.analysis = null;
    this.styleKey = 'painterly';
    this.painterLayouts = [];
  }

  configure({ analysis, duration, seed, palette, complexity, paintingStyleKey = 'trace' }) {
    this.analysis = analysis;
    this.duration = Math.max(0.001, duration);
    this.seed = seed;
    this.palette = palette;
    this.complexity = complexity;
    this.styleKey = PAINTING_STYLES[paintingStyleKey] ? paintingStyleKey : 'trace';
    this.buildPainterlyLayouts();
  }

  resize() {
    const rect = this.canvas.getBoundingClientRect();
    const width = Math.max(1, Math.floor(rect.width));
    const height = Math.max(1, Math.floor(rect.height));
    const dpr = Math.min(2, window.devicePixelRatio || 1);
    if (width === this.width && height === this.height && dpr === this.dpr) return false;
    this.width = width;
    this.height = height;
    this.dpr = dpr;
    this.canvas.width = Math.floor(width * dpr);
    this.canvas.height = Math.floor(height * dpr);
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return true;
  }

  reset() {
    if (this.styleKey === 'painterly') this.resetPainterly();
    else this.resetTrace();
  }

  resetTrace() {
    const ctx = this.ctx;
    const { width: w, height: h, palette } = this;
    ctx.save();
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    const gradient = ctx.createLinearGradient(0, 0, w, h);
    gradient.addColorStop(0, palette.bg);
    gradient.addColorStop(0.5, palette.bg2);
    gradient.addColorStop(1, palette.bg);
    ctx.fillStyle = gradient;
    ctx.fillRect(0, 0, w, h);

    ctx.globalAlpha = palette.dark ? 0.09 : 0.12;
    ctx.strokeStyle = palette.dark ? '#ffffff' : '#102424';
    ctx.lineWidth = 1;
    const spacing = Math.max(34, Math.round(w / 28));
    for (let x = spacing; x < w; x += spacing) {
      ctx.beginPath();
      ctx.moveTo(x, h * 0.08);
      ctx.lineTo(x, h * 0.92);
      ctx.stroke();
    }
    ctx.restore();
    this.prev = null;
    this.painterPrev = null;
    this.painterGroundPrev = null;
    this.lastSection = -1;
  }

  resetPainterly() {
    const ctx = this.ctx;
    const { width: w, height: h, palette } = this;
    const baseFeature = { energy: 0.32, low: 0.42, mid: 0.40, high: 0.20, centroid: 0.34 };
    ctx.save();
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);

    const base = ctx.createLinearGradient(0, h * 0.08, w, h * 0.92);
    base.addColorStop(0, palette.bg);
    base.addColorStop(0.42, palette.bg2);
    base.addColorStop(1, palette.bg);
    ctx.fillStyle = base;
    ctx.fillRect(0, 0, w, h);

    // A few broad, nearly invisible glazes establish depth before the music starts painting.
    ctx.globalCompositeOperation = palette.dark ? 'screen' : 'multiply';
    const glazeCount = 6;
    for (let i = 0; i < glazeCount; i++) {
      const cx = randAt(this.seed, i, 610) * w;
      const cy = (0.12 + randAt(this.seed, i, 611) * 0.76) * h;
      const radius = Math.max(w, h) * (0.20 + randAt(this.seed, i, 612) * 0.24);
      const radial = ctx.createRadialGradient(cx, cy, 0, cx, cy, radius);
      radial.addColorStop(0, this.color(baseFeature, cx / Math.max(1, w), i, palette.dark ? 0.045 : 0.030, (randAt(this.seed, i, 613) - 0.5) * 0.13));
      radial.addColorStop(1, this.color(baseFeature, cx / Math.max(1, w), i, 0));
      ctx.fillStyle = radial;
      ctx.fillRect(0, 0, w, h);
    }

    // Fine deterministic fibres make the surface read as paper/canvas instead of a display grid.
    ctx.globalCompositeOperation = 'source-over';
    ctx.lineCap = 'round';
    const fibreCount = Math.min(520, Math.max(180, Math.round((w * h) / 3300)));
    for (let i = 0; i < fibreCount; i++) {
      const x = randAt(this.seed, i, 620) * w;
      const y = randAt(this.seed, i, 621) * h;
      const length = 3 + randAt(this.seed, i, 622) * 22;
      const angle = (randAt(this.seed, i, 623) - 0.5) * 0.34;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + Math.cos(angle) * length, y + Math.sin(angle) * length);
      ctx.lineWidth = 0.35 + randAt(this.seed, i, 624) * 0.55;
      ctx.strokeStyle = palette.dark
        ? `rgba(255,255,255,${0.012 + randAt(this.seed, i, 625) * 0.018})`
        : `rgba(23,31,29,${0.018 + randAt(this.seed, i, 625) * 0.020})`;
      ctx.stroke();
    }

    const vignette = ctx.createRadialGradient(w * 0.5, h * 0.48, Math.min(w, h) * 0.15, w * 0.5, h * 0.48, Math.max(w, h) * 0.72);
    vignette.addColorStop(0, 'rgba(0,0,0,0)');
    vignette.addColorStop(1, palette.dark ? 'rgba(0,0,0,0.19)' : 'rgba(37,45,42,0.08)');
    ctx.fillStyle = vignette;
    ctx.fillRect(0, 0, w, h);
    ctx.restore();

    this.prev = null;
    this.painterPrev = null;
    this.painterGroundPrev = null;
    this.lastSection = -1;
  }

  color(feature, progress, sectionIndex, alpha = 1, lightShift = 0) {
    const { hue, saturation, lightness } = paletteHsl(this.palette, feature, progress, sectionIndex, this.seed, lightShift);
    return `hsla(${Math.round(hue * 360)}, ${Math.round(saturation * 100)}%, ${Math.round(lightness * 100)}%, ${alpha})`;
  }

  buildPainterlyLayouts() {
    this.painterLayouts = [];
    if (!this.analysis?.sections?.length) return;
    let y = clamp(0.42 + (randAt(this.seed, 0, 680) - 0.5) * 0.20, 0.27, 0.73);
    for (let i = 0; i < this.analysis.sections.length; i++) {
      const section = this.analysis.sections[i];
      const averages = section.averages;
      const tonalPull = (averages.low - averages.high) * 0.12;
      const randomDrift = (randAt(this.seed, i, 681) - 0.5) * (0.24 + averages.energy * 0.10);
      let endY = clamp(y + tonalPull + randomDrift, 0.22, 0.78);
      if (Math.abs(endY - y) < 0.055) {
        const direction = randAt(this.seed, i, 682) < 0.5 ? -1 : 1;
        endY = clamp(endY + direction * (0.07 + randAt(this.seed, i, 683) * 0.08), 0.22, 0.78);
      }
      const centerY = clamp((y + endY) * 0.5 + (randAt(this.seed, i, 684) - 0.5) * 0.08, 0.20, 0.80);
      const side = randAt(this.seed, i, 685) < 0.5 ? -1 : 1;
      const secondaryY = clamp(centerY + side * (0.17 + randAt(this.seed, i, 686) * 0.19), 0.10, 0.90);
      const motif = Math.floor(randAt(this.seed, i, 687) * 4);
      const angleBase = (randAt(this.seed, i, 688) - 0.5) * 0.78 + (averages.mid - 0.5) * 0.26;
      this.painterLayouts.push({
        startY: y,
        endY,
        centerY,
        secondaryY,
        groundY: clamp(0.72 + (randAt(this.seed, i, 691) - 0.5) * 0.13 - averages.low * 0.035, 0.62, 0.84),
        skyY: clamp(0.24 + (randAt(this.seed, i, 692) - 0.5) * 0.15 - averages.high * 0.025, 0.12, 0.38),
        motif,
        angleBase,
        phase: randAt(this.seed, i, 689) * TAU,
        scale: 0.84 + averages.energy * 0.38 + randAt(this.seed, i, 690) * 0.16,
      });
      y = endY;
    }
  }

  sectionWash(sectionIndex, x, feature) {
    const ctx = this.ctx;
    const w = Math.max(45, this.width * 0.055);
    const g = ctx.createLinearGradient(x - w, 0, x + w, 0);
    g.addColorStop(0, this.color(feature, x / this.width, sectionIndex, 0));
    g.addColorStop(0.5, this.color(feature, x / this.width, sectionIndex, this.palette.dark ? 0.10 : 0.07, 0.12));
    g.addColorStop(1, this.color(feature, x / this.width, sectionIndex, 0));
    ctx.save();
    ctx.globalCompositeOperation = this.palette.dark ? 'screen' : 'multiply';
    ctx.fillStyle = g;
    ctx.fillRect(x - w, 0, w * 2, this.height);
    ctx.restore();
  }

  organicBlobPath(cx, cy, rx, ry, stepIndex, channel, points = 12) {
    const ctx = this.ctx;
    const vertices = [];
    for (let i = 0; i < points; i++) {
      const angle = (i / points) * TAU;
      const radial = 0.78 + randAt(this.seed, stepIndex * 31 + i, channel) * 0.36;
      vertices.push({
        x: cx + Math.cos(angle) * rx * radial,
        y: cy + Math.sin(angle) * ry * radial,
      });
    }
    const firstMid = {
      x: (vertices[points - 1].x + vertices[0].x) * 0.5,
      y: (vertices[points - 1].y + vertices[0].y) * 0.5,
    };
    ctx.beginPath();
    ctx.moveTo(firstMid.x, firstMid.y);
    for (let i = 0; i < points; i++) {
      const current = vertices[i];
      const next = vertices[(i + 1) % points];
      ctx.quadraticCurveTo(current.x, current.y, (current.x + next.x) * 0.5, (current.y + next.y) * 0.5);
    }
    ctx.closePath();
  }

  drawOrganicBlob(cx, cy, rx, ry, feature, progress, sectionIndex, alpha, stepIndex, channel) {
    const ctx = this.ctx;
    const layers = this.complexity > 1.15 ? 3 : 2;
    for (let layer = 0; layer < layers; layer++) {
      const jitterX = (randAt(this.seed, stepIndex + layer, channel + 1) - 0.5) * rx * 0.20;
      const jitterY = (randAt(this.seed, stepIndex + layer, channel + 2) - 0.5) * ry * 0.20;
      const scale = 0.78 + randAt(this.seed, stepIndex + layer, channel + 3) * 0.28;
      this.organicBlobPath(cx + jitterX, cy + jitterY, rx * scale, ry * scale, stepIndex + layer * 97, channel + 10 + layer * 17, 11 + layer);
      ctx.fillStyle = this.color(
        feature,
        progress,
        sectionIndex,
        alpha * (layer === 0 ? 0.64 : 0.38),
        (randAt(this.seed, stepIndex + layer, channel + 4) - 0.5) * 0.16,
      );
      ctx.fill();
    }
  }

  drawBristleStroke({ x1, y1, x2, y2, width, curve, feature, progress, sectionIndex, alpha, stepIndex, channel }) {
    const ctx = this.ctx;
    const dx = x2 - x1;
    const dy = y2 - y1;
    const length = Math.max(0.001, Math.hypot(dx, dy));
    const nx = -dy / length;
    const ny = dx / length;
    const cx1 = lerp(x1, x2, 0.34) + nx * curve;
    const cy1 = lerp(y1, y2, 0.34) + ny * curve;
    const cx2 = lerp(x1, x2, 0.70) - nx * curve * 0.45;
    const cy2 = lerp(y1, y2, 0.70) - ny * curve * 0.45;

    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.bezierCurveTo(cx1, cy1, cx2, cy2, x2, y2);
    ctx.lineWidth = width;
    ctx.strokeStyle = this.color(feature, progress, sectionIndex, alpha * 0.52, -0.06);
    ctx.stroke();

    const bristles = Math.max(3, Math.round(2.7 + this.complexity * 2.2));
    for (let i = 0; i < bristles; i++) {
      const offset = (randAt(this.seed, stepIndex * 13 + i, channel) - 0.5) * width * 0.92;
      const startGap = randAt(this.seed, stepIndex * 17 + i, channel + 1) * 0.16;
      const endGap = randAt(this.seed, stepIndex * 19 + i, channel + 2) * 0.12;
      const sx = lerp(x1, x2, startGap) + nx * offset;
      const sy = lerp(y1, y2, startGap) + ny * offset;
      const ex = lerp(x1, x2, 1 - endGap) + nx * offset * 0.55;
      const ey = lerp(y1, y2, 1 - endGap) + ny * offset * 0.55;
      const lightShift = (randAt(this.seed, stepIndex * 23 + i, channel + 3) - 0.5) * 0.22;
      ctx.beginPath();
      ctx.moveTo(sx, sy);
      ctx.bezierCurveTo(
        lerp(sx, ex, 0.34) + nx * curve * 0.72,
        lerp(sy, ey, 0.34) + ny * curve * 0.72,
        lerp(sx, ex, 0.70) - nx * curve * 0.28,
        lerp(sy, ey, 0.70) - ny * curve * 0.28,
        ex,
        ey,
      );
      ctx.lineWidth = Math.max(0.45, width * (0.055 + randAt(this.seed, stepIndex * 29 + i, channel + 4) * 0.11));
      ctx.strokeStyle = this.color(feature, progress, sectionIndex, alpha * (0.28 + randAt(this.seed, stepIndex * 31 + i, channel + 5) * 0.34), lightShift);
      ctx.stroke();
    }
  }

  drawKnifeMark(x, y, angle, length, width, feature, progress, sectionIndex, stepIndex, channel) {
    const ctx = this.ctx;
    const c = Math.cos(angle);
    const s = Math.sin(angle);
    const nx = -s;
    const ny = c;
    const backX = x - c * length;
    const backY = y - s * length;
    const skew = (randAt(this.seed, stepIndex, channel) - 0.5) * width;
    ctx.beginPath();
    ctx.moveTo(backX + nx * width * 0.16, backY + ny * width * 0.16);
    ctx.bezierCurveTo(
      lerp(backX, x, 0.34) + nx * (width + skew),
      lerp(backY, y, 0.34) + ny * (width + skew),
      lerp(backX, x, 0.76) + nx * width * 0.42,
      lerp(backY, y, 0.76) + ny * width * 0.42,
      x,
      y,
    );
    ctx.bezierCurveTo(
      lerp(x, backX, 0.30) - nx * width * 0.58,
      lerp(y, backY, 0.30) - ny * width * 0.58,
      lerp(x, backX, 0.76) - nx * width * 0.20,
      lerp(y, backY, 0.76) - ny * width * 0.20,
      backX + nx * width * 0.16,
      backY + ny * width * 0.16,
    );
    ctx.closePath();
    ctx.fillStyle = this.color(feature, progress, sectionIndex, this.palette.dark ? 0.58 : 0.42, 0.10);
    ctx.fill();

    ctx.beginPath();
    ctx.moveTo(backX + nx * width * 0.13, backY + ny * width * 0.13);
    ctx.quadraticCurveTo(lerp(backX, x, 0.56) + nx * width * 0.25, lerp(backY, y, 0.56) + ny * width * 0.25, x, y);
    ctx.lineWidth = Math.max(0.55, width * 0.10);
    ctx.strokeStyle = this.color(feature, progress, sectionIndex, 0.78, 0.22);
    ctx.stroke();
  }

  drawPigmentMassSegment(previous, x, y, feature, progress, sectionIndex, stepIndex) {
    const ctx = this.ctx;
    const midX = (previous.x + x) * 0.5;
    const edgeY = (previous.y + y) * 0.5 + (randAt(this.seed, stepIndex, 795) - 0.5) * (4 + feature.low * 12);

    // Overlapping broad strokes accumulate like an underpainted ground without digital vertical seams.
    ctx.beginPath();
    ctx.moveTo(previous.x, previous.y);
    ctx.quadraticCurveTo(midX, edgeY, x, y);
    ctx.lineWidth = this.height * (0.055 + feature.low * 0.095 + feature.energy * 0.028);
    ctx.strokeStyle = this.color(feature, progress, sectionIndex, this.palette.dark ? 0.075 : 0.052, -0.12);
    ctx.stroke();

    if (stepIndex % 3 === 0) {
      ctx.beginPath();
      ctx.moveTo(previous.x, previous.y);
      ctx.quadraticCurveTo(midX, edgeY, x, y);
      ctx.lineWidth = 1.0 + feature.low * 3.1 + feature.energy * 1.4;
      ctx.strokeStyle = this.color(feature, progress, sectionIndex, this.palette.dark ? 0.24 : 0.17, -0.02);
      ctx.stroke();
    }
  }
  painterlyPosition(feature, progress, stepIndex) {
    const h = this.height;
    const layout = this.painterLayouts[feature.sectionIndex] || {
      startY: 0.46,
      endY: 0.54,
      centerY: 0.50,
      secondaryY: 0.72,
      groundY: 0.74,
      skyY: 0.24,
      motif: 0,
      angleBase: 0,
      phase: 0,
      scale: 1,
    };
    const local = feature.sectionPhase;
    const eased = smoothstep(0, 1, local);
    const primary = lerp(layout.startY, layout.endY, eased) * h;
    const secondary = layout.secondaryY * h;
    const pitchNorm = feature.dominant > 1
      ? clamp((Math.log2(Math.max(55, feature.dominant)) - Math.log2(55)) / 6)
      : feature.centroid;
    const pitchShift = (0.5 - pitchNorm) * h * 0.21;
    const noise = (valueNoise1D(progress * (8 + this.complexity * 3.2), this.seed, 704 + feature.sectionIndex) - 0.5) * h * 0.14;
    let y = primary + pitchShift * 0.46 + noise;
    let angle = layout.angleBase;

    if (layout.motif === 0) {
      // Landscape: pitch chooses one of several painterly planes rather than tracing one centre line.
      const upper = clamp(Math.min(primary, secondary) - h * 0.08, h * 0.12, h * 0.42);
      const middle = clamp(layout.centerY * h, h * 0.28, h * 0.72);
      const lower = clamp(Math.max(primary, secondary) + h * 0.08, h * 0.54, h * 0.86);
      if (pitchNorm > 0.64) y = upper + noise * 0.30;
      else if (pitchNorm < 0.34) y = lower + noise * 0.30;
      else y = middle + noise * 0.42;
      y += Math.sin(local * Math.PI * 1.4 + layout.phase) * h * 0.025;
      angle += (feature.mid - 0.5) * 0.34;
    } else if (layout.motif === 1) {
      // Bloom: gestures orbit a compositional centre instead of tracing a graph.
      const orbit = local * TAU * (0.72 + layout.scale * 0.34) + layout.phase;
      const radius = h * (0.045 + feature.energy * 0.15) * layout.scale;
      y = layout.centerY * h + Math.sin(orbit) * radius + pitchShift * 0.30 + noise * 0.48;
      angle = clamp(Math.cos(orbit) * 0.92 + (feature.centroid - 0.5) * 0.35, -1.15, 1.15);
    } else if (layout.motif === 2) {
      // Veils: alternating vertical fields create figure/ground relationships.
      const weave = 0.5 + 0.5 * Math.sin(local * Math.PI * (2.4 + layout.scale) + layout.phase);
      y = lerp(primary, secondary, weave * (0.55 + feature.high * 0.28)) + pitchShift * 0.42 + noise * 0.40;
      angle = clamp(layout.angleBase + (weave - 0.5) * 1.36, -1.20, 1.20);
    } else {
      // Archipelago: grouped marks form islands, keeping deliberate empty space between them.
      const cluster = Math.floor(local * (3 + Math.round(layout.scale * 2)));
      const chooseSecondary = randAt(this.seed, feature.sectionIndex * 17 + cluster, 706) > 0.54;
      const clusterY = chooseSecondary ? secondary : primary;
      y = clusterY + (randAt(this.seed, feature.sectionIndex * 29 + cluster, 707) - 0.5) * h * 0.13 + pitchShift * 0.38;
      angle = clamp(layout.angleBase + (randAt(this.seed, feature.sectionIndex * 31 + cluster, 708) - 0.5) * 0.92, -1.08, 1.08);
    }

    return {
      y: clamp(y, h * 0.09, h * 0.91),
      secondaryY: clamp(secondary, h * 0.08, h * 0.92),
      angle,
      layout,
    };
  }

  generate(time, feature, stepIndex) {
    if (this.styleKey === 'painterly') this.generatePainterly(time, feature, stepIndex);
    else this.generateTrace(time, feature, stepIndex);
  }

  generateTrace(time, feature, stepIndex) {
    const ctx = this.ctx;
    const w = this.width;
    const h = this.height;
    if (w < 2 || h < 2) return;
    const marginX = Math.max(24, w * 0.035);
    const progress = clamp(time / this.duration);
    const x = marginX + progress * (w - marginX * 2);
    const section = this.analysis.sections[feature.sectionIndex] || this.analysis.sections[0];
    const local = feature.sectionPhase;
    const motif = (feature.sectionIndex + Math.floor(randAt(this.seed, feature.sectionIndex, 21) * 4)) % 4;
    const n1 = valueNoise1D(progress * (5.5 + this.complexity * 2.2), this.seed, 4) - 0.5;
    const n2 = valueNoise1D(progress * (11 + this.complexity * 3), this.seed, 7) - 0.5;
    const sectionBias = (section.averages.low - section.averages.high) * h * 0.12;
    let motifCurve = 0;
    if (motif === 0) motifCurve = Math.sin(local * Math.PI) * (feature.sectionIndex % 2 ? -1 : 1);
    if (motif === 1) motifCurve = Math.sin(local * TAU * 1.25 + feature.sectionIndex) * 0.55;
    if (motif === 2) motifCurve = (local - 0.5) * (feature.sectionIndex % 2 ? -1 : 1);
    if (motif === 3) motifCurve = Math.sin(local * Math.PI * 3) * (0.25 + 0.4 * feature.mid);
    const targetY = h * 0.5 + sectionBias + motifCurve * h * 0.115 + n1 * h * 0.19 + (feature.low - 0.5) * h * 0.13;
    const y = this.prev ? lerp(this.prev.y, targetY, 0.18 + feature.onset * 0.12) : targetY;
    const band = (5 + feature.energy * 22 + feature.low * 16) * (0.72 + this.complexity * 0.32);
    const tilt = (feature.mid - 0.5) * 0.9 + n2 * 0.6;

    if (feature.sectionIndex !== this.lastSection) {
      this.sectionWash(feature.sectionIndex, x, feature);
      this.lastSection = feature.sectionIndex;
    }

    if (!this.prev) {
      this.prev = { x, y, band, tilt };
      return;
    }

    const prev = this.prev;
    const color = this.color(feature, progress, feature.sectionIndex, this.palette.dark ? 0.45 : 0.32);
    const colorBright = this.color(feature, progress, feature.sectionIndex, this.palette.dark ? 0.84 : 0.72, 0.10);
    const colorDim = this.color(feature, progress, feature.sectionIndex, this.palette.dark ? 0.20 : 0.14, -0.10);

    ctx.save();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.globalCompositeOperation = this.palette.dark ? 'screen' : 'multiply';

    // The low-frequency body: a continuous ribbon that makes the whole piece read as one composition.
    ctx.beginPath();
    ctx.moveTo(prev.x, prev.y - prev.band * (0.6 + prev.tilt * 0.08));
    ctx.quadraticCurveTo((prev.x + x) / 2, (prev.y + y) / 2 - (prev.band + band) * 0.55, x, y - band * (0.6 + tilt * 0.08));
    ctx.lineTo(x, y + band * (0.6 - tilt * 0.08));
    ctx.quadraticCurveTo((prev.x + x) / 2, (prev.y + y) / 2 + (prev.band + band) * 0.55, prev.x, prev.y + prev.band * (0.6 - prev.tilt * 0.08));
    ctx.closePath();
    ctx.fillStyle = colorDim;
    ctx.fill();

    // Main melodic stroke.
    ctx.beginPath();
    ctx.moveTo(prev.x, prev.y);
    const bow = (feature.mid - 0.5) * h * 0.02 + n2 * h * 0.012;
    ctx.quadraticCurveTo((prev.x + x) / 2, (prev.y + y) / 2 + bow, x, y);
    ctx.lineWidth = Math.max(1.2, 1.1 + feature.energy * 4.4 + feature.mid * 2.5);
    ctx.strokeStyle = colorBright;
    ctx.stroke();

    // A treble thread mirrors the melody and adds readable fine structure.
    const trebleOffset = 13 + feature.high * 42;
    const side = stepIndex % 2 ? 1 : -1;
    ctx.beginPath();
    ctx.moveTo(prev.x, prev.y + side * (11 + feature.high * 20));
    ctx.quadraticCurveTo((prev.x + x) / 2, (prev.y + y) / 2 + side * trebleOffset * 1.2, x, y + side * trebleOffset);
    ctx.lineWidth = 0.55 + feature.high * 1.35;
    ctx.strokeStyle = color;
    ctx.stroke();

    // Strong onsets become calligraphic leaves/petals; they are permanent but spatially disciplined.
    const accentChance = feature.onset * (0.76 + this.complexity * 0.28) + feature.high * 0.05;
    if (accentChance > 0.34 && randAt(this.seed, stepIndex, 31) < accentChance) {
      const direction = randAt(this.seed, stepIndex, 32) < 0.5 ? -1 : 1;
      const length = 10 + feature.onset * 45 + feature.high * 28;
      const width = 2 + feature.energy * 8;
      const angle = direction * (0.35 + randAt(this.seed, stepIndex, 33) * 0.9);
      const ex = x + Math.cos(angle) * length;
      const ey = y + Math.sin(angle) * length;
      ctx.shadowColor = colorBright;
      ctx.shadowBlur = this.palette.dark ? 10 + feature.onset * 16 : 2;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.bezierCurveTo(x + length * 0.28, y + direction * width * 2.5, ex - length * 0.2, ey - direction * width, ex, ey);
      ctx.bezierCurveTo(ex - length * 0.32, ey + direction * width * 1.5, x + length * 0.16, y - direction * width * 2.2, x, y);
      ctx.fillStyle = this.color(feature, progress, feature.sectionIndex, this.palette.dark ? 0.50 : 0.34, 0.14);
      ctx.fill();
      ctx.shadowBlur = 0;

      ctx.beginPath();
      ctx.arc(x, y, 1.5 + feature.onset * 4.2, 0, TAU);
      ctx.fillStyle = colorBright;
      ctx.fill();
    }

    // Very high frequencies produce sparse dust, never a full-screen particle storm.
    if (feature.high > 0.58 && randAt(this.seed, stepIndex, 40) < feature.high * 0.36 * this.complexity) {
      const count = 1 + Math.floor(feature.high * 3);
      for (let i = 0; i < count; i++) {
        const rx = (randAt(this.seed, stepIndex, 41 + i) - 0.5) * 18;
        const ry = (randAt(this.seed, stepIndex, 47 + i) - 0.5) * (35 + feature.high * 70);
        ctx.beginPath();
        ctx.arc(x + rx, y + ry, 0.5 + randAt(this.seed, stepIndex, 55 + i) * 1.7, 0, TAU);
        ctx.fillStyle = this.color(feature, progress, feature.sectionIndex, 0.38 + feature.high * 0.28, 0.18);
        ctx.fill();
      }
    }

    ctx.restore();
    this.prev = { x, y, band, tilt };
  }

  generatePainterly(time, feature, stepIndex) {
    const ctx = this.ctx;
    const w = this.width;
    const h = this.height;
    if (w < 2 || h < 2) return;
    const marginX = Math.max(26, w * 0.04);
    const progress = clamp(time / this.duration);
    const x = marginX + progress * (w - marginX * 2);
    const position = this.painterlyPosition(feature, progress, stepIndex);
    const y = position.y;
    const layout = position.layout;
    const composite = this.palette.dark ? 'screen' : 'multiply';

    ctx.save();
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';
    ctx.globalCompositeOperation = composite;

    if (feature.sectionIndex !== this.lastSection) {
      const openingWidth = Math.max(24, w * (0.035 + feature.energy * 0.025));
      const openingHeight = h * (0.14 + layout.scale * 0.12);
      this.drawOrganicBlob(
        x - openingWidth * 0.36,
        lerp(y, position.secondaryY, 0.34),
        openingWidth,
        openingHeight,
        feature,
        progress,
        feature.sectionIndex,
        this.palette.dark ? 0.10 : 0.075,
        stepIndex,
        720,
      );
      this.lastSection = feature.sectionIndex;
      this.painterPrev = null;
      this.painterGroundPrev = null;
    }

    // Bass continuously accumulates as a lower pigment mass, making the result read as a painted field.
    const groundNoise = (valueNoise1D(progress * (5.5 + this.complexity * 2), this.seed, 790 + feature.sectionIndex) - 0.5) * h * 0.055;
    const groundY = clamp(layout.groundY * h - feature.low * h * 0.075 + groundNoise, h * 0.53, h * 0.88);
    if (this.painterGroundPrev && this.painterGroundPrev.sectionIndex === feature.sectionIndex) {
      this.drawPigmentMassSegment(this.painterGroundPrev, x, groundY, feature, progress, feature.sectionIndex, stepIndex);
    }
    this.painterGroundPrev = { x, y: groundY, sectionIndex: feature.sectionIndex };

    // Treble occasionally opens a separate sky field, creating figure/ground rather than one plotted path.
    const skyStride = Math.max(11, Math.round(24 - this.complexity * 7));
    if (feature.high > 0.28 && stepIndex % skyStride === 0) {
      const skyY = layout.skyY * h + (randAt(this.seed, stepIndex, 791) - 0.5) * h * 0.12;
      this.drawOrganicBlob(
        Math.max(marginX, x - (24 + feature.high * 38) * 0.44),
        skyY,
        24 + feature.high * 38,
        18 + feature.high * 40,
        feature,
        progress,
        feature.sectionIndex,
        this.palette.dark ? 0.075 : 0.052,
        stepIndex,
        792,
      );
    }

    // Slow colour fields are the underpainting. They build atmosphere without becoming a waveform.
    const washStride = Math.max(8, Math.round(19 - this.complexity * 6));
    if (stepIndex % washStride === 0) {
      const rx = 26 + feature.energy * 54 + feature.low * 28;
      const ry = 30 + feature.mid * 58 + feature.high * 24;
      const washY = lerp(y, position.secondaryY, randAt(this.seed, stepIndex, 721) * (0.34 + feature.high * 0.38));
      this.drawOrganicBlob(
        Math.max(marginX, x - rx * 0.44),
        washY,
        rx,
        ry,
        feature,
        progress,
        feature.sectionIndex,
        this.palette.dark ? 0.105 : 0.072,
        stepIndex,
        722,
      );
    }

    // Bass and overall energy lay down broad, dry-brush masses.
    const massStride = Math.max(2, Math.round(5.2 - this.complexity * 2.1));
    if (stepIndex % massStride === 0 && (feature.low > 0.20 || feature.energy > 0.30)) {
      const length = (18 + feature.low * 58 + feature.energy * 22) * layout.scale;
      const width = (5.0 + feature.low * 20 + feature.energy * 9) * (0.78 + this.complexity * 0.24);
      const angle = clamp((randAt(this.seed, stepIndex, 730) - 0.5) * 0.46 + (feature.mid - 0.5) * 0.16, -0.48, 0.48);
      this.drawBristleStroke({
        x1: Math.max(marginX, x - Math.cos(angle) * length),
        y1: groundY - Math.sin(angle) * length,
        x2: x,
        y2: groundY,
        width,
        curve: (randAt(this.seed, stepIndex, 731) - 0.5) * width * 1.8,
        feature,
        progress,
        sectionIndex: feature.sectionIndex,
        alpha: this.palette.dark ? 0.58 : 0.46,
        stepIndex,
        channel: 732,
      });
    }

    // Midrange and pitch become discrete painterly gestures, not a single connected centre line.
    const gestureStride = Math.max(1, Math.round(3.4 - this.complexity * 1.55));
    if (stepIndex % gestureStride === 0) {
      const length = 9 + feature.mid * 28 + feature.onset * 18 + feature.centroid * 7;
      const width = 2.2 + feature.energy * 7.0 + feature.mid * 4.6;
      const angle = clamp(position.angle + (feature.centroid - 0.5) * 0.42 + (randAt(this.seed, stepIndex, 740) - 0.5) * 0.34, -1.22, 1.22);
      this.drawBristleStroke({
        x1: Math.max(marginX, x - Math.cos(angle) * length),
        y1: y - Math.sin(angle) * length,
        x2: x,
        y2: y,
        width,
        curve: (feature.mid - 0.5) * width * 1.8 + (randAt(this.seed, stepIndex, 741) - 0.5) * width,
        feature,
        progress,
        sectionIndex: feature.sectionIndex,
        alpha: this.palette.dark ? 0.78 : 0.62,
        stepIndex,
        channel: 742,
      });

      // Harmonic echoes occupy a second compositional area and keep the canvas from collapsing into one path.
      const echoChance = (0.08 + feature.mid * 0.18 + feature.high * 0.08) * this.complexity;
      if (randAt(this.seed, stepIndex, 748) < echoChance) {
        const echoY = lerp(position.secondaryY, y, 0.18 + randAt(this.seed, stepIndex, 749) * 0.28);
        const echoLength = length * (0.46 + randAt(this.seed, stepIndex, 750) * 0.30);
        const echoAngle = clamp(-angle * 0.62 + (randAt(this.seed, stepIndex, 751) - 0.5) * 0.34, -1.18, 1.18);
        this.drawBristleStroke({
          x1: Math.max(marginX, x - Math.cos(echoAngle) * echoLength),
          y1: echoY - Math.sin(echoAngle) * echoLength,
          x2: x,
          y2: echoY,
          width: Math.max(0.9, width * 0.48),
          curve: (randAt(this.seed, stepIndex, 752) - 0.5) * width,
          feature,
          progress,
          sectionIndex: feature.sectionIndex,
          alpha: this.palette.dark ? 0.40 : 0.30,
          stepIndex,
          channel: 753,
        });
      }
    }

    // Strong transients read as decisive palette-knife/calligraphic marks.
    const accentSignal = feature.onset * (0.82 + this.complexity * 0.20) + feature.energy * 0.09;
    if (accentSignal > 0.31 && randAt(this.seed, stepIndex, 760) < accentSignal) {
      const direction = randAt(this.seed, stepIndex, 761) < 0.5 ? -1 : 1;
      const angle = clamp(position.angle + direction * (0.32 + randAt(this.seed, stepIndex, 762) * 0.72), -1.30, 1.30);
      this.drawKnifeMark(
        x,
        y,
        angle,
        12 + feature.onset * 46 + feature.high * 16,
        2.4 + feature.energy * 8.8,
        feature,
        progress,
        feature.sectionIndex,
        stepIndex,
        763,
      );
    }

    // High frequencies add restrained spatter and dry-brush flecks around existing masses.
    if (feature.high > 0.48 && randAt(this.seed, stepIndex, 770) < feature.high * 0.34 * this.complexity) {
      const count = 1 + Math.floor(feature.high * (2.2 + this.complexity));
      for (let i = 0; i < count; i++) {
        const spreadX = 8 + feature.high * 28;
        const spreadY = 18 + feature.high * 64;
        const px = x - randAt(this.seed, stepIndex * 11 + i, 771) * spreadX;
        const fleckAnchor = randAt(this.seed, stepIndex * 19 + i, 774) < 0.58 ? layout.skyY * h : position.secondaryY;
        const py = fleckAnchor + (randAt(this.seed, stepIndex * 13 + i, 772) - 0.5) * spreadY;
        const radius = 0.45 + randAt(this.seed, stepIndex * 17 + i, 773) * (1.2 + feature.high * 1.7);
        ctx.beginPath();
        ctx.arc(px, py, radius, 0, TAU);
        ctx.fillStyle = this.color(feature, progress, feature.sectionIndex, 0.30 + feature.high * 0.30, 0.16);
        ctx.fill();
      }
    }


    ctx.restore();
    this.painterPrev = { x, y, sectionIndex: feature.sectionIndex };
  }
}
class Sculpture3D {
  constructor(container) {
    this.container = container;
    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(47, 1, 0.05, 5000);
    this.camera.position.set(18, 14, 28);
    this.renderer = new THREE.WebGLRenderer({ antialias: true, preserveDrawingBuffer: true, powerPreference: 'high-performance' });
    this.renderer.setPixelRatio(Math.min(2, window.devicePixelRatio || 1));
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 1.02;
    this.container.appendChild(this.renderer.domElement);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.065;
    this.controls.minDistance = 2;
    this.controls.maxDistance = 2500;
    this.controls.addEventListener('start', () => { this.cameraMode = 'manual'; updateCameraModeUI(); });

    this.root = new THREE.Group();
    this.scene.add(this.root);
    this.hemi = new THREE.HemisphereLight(0xbddfff, 0x241928, 0.98);
    this.scene.add(this.hemi);
    this.keyLight = new THREE.DirectionalLight(0xffffff, 1.72);
    this.keyLight.position.set(10, 18, 12);
    this.scene.add(this.keyLight);
    this.rimLight = new THREE.DirectionalLight(0x8aa8ff, 1.48);
    this.rimLight.position.set(-12, 4, -10);
    this.scene.add(this.rimLight);
    this.tipLight = new THREE.PointLight(0x77ffff, 2.2, 28, 2);
    this.scene.add(this.tipLight);

    const tipMat = new THREE.MeshBasicMaterial({ color: 0xffffff, transparent: true, opacity: 0.86 });
    this.tip = new THREE.Mesh(new THREE.SphereGeometry(0.22, 16, 12), tipMat);
    this.scene.add(this.tip);

    this.grid = new THREE.GridHelper(240, 48, 0x65748a, 0x263244);
    this.grid.material.transparent = true;
    this.grid.material.opacity = 0.12;
    this.grid.position.y = -10;
    this.scene.add(this.grid);

    this.cameraMode = 'follow';
    this.palette = PALETTES.aurora;
    this.seed = 1;
    this.complexity = 1;
    this.analysis = null;
    this.duration = 1;
    this.maxSteps = 1000;
    this.totalLength = 80;
    this.prevPoint = new THREE.Vector3(0, 0, 0);
    this.point = new THREE.Vector3(0, 0, 0);
    this.velocity = new THREE.Vector3(1, 0, 0);
    this.tangent = new THREE.Vector3(1, 0, 0);
    this.bounds = new THREE.Box3();
    this.hasBounds = false;
    this.trunkCount = 0;
    this.branchCount = 0;
    this.nodeCount = 0;
    this.sparkCount = 0;
    this.sectionMeshes = [];
    this.lastSection = -1;
    this.lastBranchStep = -999;
    this.lastNodeStep = -999;
    this.tmp = {
      matrix: new THREE.Matrix4(),
      quaternion: new THREE.Quaternion(),
      scale: new THREE.Vector3(),
      midpoint: new THREE.Vector3(),
      direction: new THREE.Vector3(),
      basis1: new THREE.Vector3(),
      basis2: new THREE.Vector3(),
      endpoint: new THREE.Vector3(),
      color: new THREE.Color(),
      color2: new THREE.Color(),
    };
  }

  configure({ analysis, duration, seed, palette, complexity, stepDuration }) {
    this.analysis = analysis;
    this.duration = Math.max(0.001, duration);
    this.seed = seed;
    this.palette = palette;
    this.complexity = complexity;
    this.maxSteps = Math.ceil(duration / stepDuration) + 16;
    this.totalLength = clamp(duration * 0.47, 48, 260);
    this.createInstances();
    this.applyPalette();
  }

  applyPalette() {
    const p = this.palette;
    this.scene.background = new THREE.Color(p.bg);
    this.scene.fog = null;
    this.grid.material.color.set(p.dark ? 0x3a4a62 : 0x60706c);
    this.grid.material.opacity = p.dark ? 0.12 : 0.16;
    this.hemi.color.set(p.dark ? 0xc8e7ff : 0xffffff);
    this.hemi.groundColor.set(p.dark ? 0x24152d : 0x78827c);
    this.rimLight.color.set(p.accent);
    this.tipLight.color.set(p.accent);
  }

  disposeObject(object) {
    if (!object) return;
    object.traverse((child) => {
      if (child.geometry) child.geometry.dispose();
      if (child.material) {
        if (Array.isArray(child.material)) child.material.forEach((m) => m.dispose());
        else child.material.dispose();
      }
    });
  }

  clearRoot() {
    while (this.root.children.length) {
      const child = this.root.children[this.root.children.length - 1];
      this.root.remove(child);
      this.disposeObject(child);
    }
    this.sectionMeshes = [];
  }

  createInstances() {
    this.clearRoot();
    const trunkGeometry = new THREE.CylinderGeometry(1, 1, 1, 9, 1, false);
    const branchGeometry = new THREE.CylinderGeometry(0.82, 1, 1, 7, 1, false);
    const nodeGeometry = new THREE.OctahedronGeometry(1, 0);
    const sparkGeometry = new THREE.TetrahedronGeometry(1, 0);
    const trunkMaterial = new THREE.MeshStandardMaterial({ roughness: 0.46, metalness: 0.10 });
    const branchMaterial = new THREE.MeshStandardMaterial({ roughness: 0.55, metalness: 0.08, transparent: true, opacity: 0.92 });
    const nodeMaterial = new THREE.MeshStandardMaterial({ roughness: 0.36, metalness: 0.22, flatShading: true });
    const sparkMaterial = new THREE.MeshStandardMaterial({ roughness: 0.24, metalness: 0.30, transparent: true, opacity: 0.84, flatShading: true });

    this.trunk = new THREE.InstancedMesh(trunkGeometry, trunkMaterial, this.maxSteps);
    this.branches = new THREE.InstancedMesh(branchGeometry, branchMaterial, Math.ceil(this.maxSteps * 1.55));
    this.nodes = new THREE.InstancedMesh(nodeGeometry, nodeMaterial, Math.ceil(this.maxSteps * 0.72));
    this.sparks = new THREE.InstancedMesh(sparkGeometry, sparkMaterial, Math.ceil(this.maxSteps * 1.1));
    for (const mesh of [this.trunk, this.branches, this.nodes, this.sparks]) {
      mesh.count = 0;
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      mesh.frustumCulled = false;
      this.root.add(mesh);
    }
  }

  resize() {
    const rect = this.container.getBoundingClientRect();
    const width = Math.max(1, rect.width);
    const height = Math.max(1, rect.height);
    const size = this.renderer.getSize(new THREE.Vector2());
    if (size.x === width && size.y === height) return false;
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height, false);
    return true;
  }

  reset() {
    this.trunkCount = 0;
    this.branchCount = 0;
    this.nodeCount = 0;
    this.sparkCount = 0;
    if (this.trunk) this.trunk.count = 0;
    if (this.branches) this.branches.count = 0;
    if (this.nodes) this.nodes.count = 0;
    if (this.sparks) this.sparks.count = 0;
    for (const mesh of this.sectionMeshes) {
      this.root.remove(mesh);
      this.disposeObject(mesh);
    }
    this.sectionMeshes.length = 0;
    this.lastSection = -1;
    this.lastBranchStep = -999;
    this.lastNodeStep = -999;
    this.prevPoint.set(0, 0, 0);
    this.point.set(0, 0, 0);
    this.velocity.set(this.totalLength / Math.max(1, this.maxSteps), 0, 0);
    this.tangent.set(1, 0, 0);
    this.bounds.makeEmpty();
    this.hasBounds = false;
    this.tip.position.copy(this.point);
    this.tip.visible = true;
    this.expandBounds(this.point);
    this.camera.position.set(18, 14, 28);
    this.controls.target.set(0, 0, 0);
    this.cameraMode = 'follow';
    updateCameraModeUI();
  }

  colorFor(feature, progress, lightShift = 0) {
    const hsl = paletteHsl(this.palette, feature, progress, feature.sectionIndex, this.seed, lightShift);
    return this.tmp.color.setHSL(hsl.hue, hsl.saturation, hsl.lightness);
  }

  setCylinderInstance(mesh, index, a, b, radiusA, radiusB, color) {
    const { direction, midpoint, quaternion, scale, matrix } = this.tmp;
    direction.subVectors(b, a);
    const length = Math.max(0.0001, direction.length());
    direction.normalize();
    midpoint.addVectors(a, b).multiplyScalar(0.5);
    quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
    const radius = Math.max(0.01, (radiusA + radiusB) * 0.5);
    scale.set(radius, length, radius);
    matrix.compose(midpoint, quaternion, scale);
    mesh.setMatrixAt(index, matrix);
    mesh.setColorAt(index, color);
  }

  setShapeInstance(mesh, index, position, scaleValue, rotationSeed, color, stretch = 1) {
    const { quaternion, scale, matrix } = this.tmp;
    const euler = new THREE.Euler(
      randAt(this.seed, rotationSeed, 70) * TAU,
      randAt(this.seed, rotationSeed, 71) * TAU,
      randAt(this.seed, rotationSeed, 72) * TAU,
    );
    quaternion.setFromEuler(euler);
    scale.set(scaleValue, scaleValue * stretch, scaleValue * (2 - stretch));
    matrix.compose(position, quaternion, scale);
    mesh.setMatrixAt(index, matrix);
    mesh.setColorAt(index, color);
  }

  expandBounds(point) {
    this.bounds.expandByPoint(point);
    this.hasBounds = true;
  }

  createSectionRing(feature, stepIndex) {
    const radius = 0.72 + feature.energy * 1.35 + this.complexity * 0.20;
    const geometry = new THREE.TorusGeometry(radius, 0.022 + feature.high * 0.032, 7, 42);
    const color = this.colorFor(feature, stepIndex / this.maxSteps, 0.15).clone();
    const material = new THREE.MeshStandardMaterial({ color, emissive: color.clone().multiplyScalar(0.22), roughness: 0.38, metalness: 0.25, transparent: true, opacity: 0.48 });
    const ring = new THREE.Mesh(geometry, material);
    ring.position.copy(this.point);
    ring.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), this.tangent.clone().normalize());
    this.root.add(ring);
    this.sectionMeshes.push(ring);
    const extent = new THREE.Vector3(radius, radius, radius);
    this.expandBounds(this.point.clone().add(extent));
    this.expandBounds(this.point.clone().sub(extent));
  }

  generate(time, feature, stepIndex) {
    if (!this.trunk || this.trunkCount >= this.trunk.instanceMatrix.count) return;
    const progress = clamp(time / this.duration);
    const dx = this.totalLength / Math.max(1, this.maxSteps - 1);
    const motif = (feature.sectionIndex + Math.floor(randAt(this.seed, feature.sectionIndex, 112) * 4)) % 4;
    const phase = progress * TAU * (2.2 + motif * 0.37) + feature.sectionIndex * 0.91;
    const noiseY = valueNoise1D(progress * (8 + this.complexity * 4), this.seed, 120) - 0.5;
    const noiseZ = valueNoise1D(progress * (10 + this.complexity * 5), this.seed, 121) - 0.5;
    let motifY = Math.sin(phase);
    let motifZ = Math.cos(phase * 0.72 + 0.6);
    if (motif === 1) { motifY = Math.sin(phase * 0.55) * Math.cos(phase * 0.18); motifZ = Math.sin(phase * 1.15); }
    if (motif === 2) { motifY = Math.sin(phase * 1.8) * 0.5; motifZ = Math.cos(phase * 1.3) * 0.72; }
    if (motif === 3) { motifY = Math.sin(phase * 0.35) + Math.sin(phase * 1.7) * 0.22; motifZ = Math.cos(phase * 0.48) - Math.sin(phase * 1.4) * 0.22; }

    const desiredVy = dx * ((feature.low - 0.45) * 0.95 + motifY * (0.18 + feature.mid * 0.24) + noiseY * (0.42 + this.complexity * 0.2));
    const desiredVz = dx * ((feature.high - feature.mid) * 0.78 + motifZ * (0.16 + feature.centroid * 0.28) + noiseZ * (0.42 + this.complexity * 0.22));
    this.velocity.x = dx;
    this.velocity.y = lerp(this.velocity.y, desiredVy, 0.14 + feature.onset * 0.05);
    this.velocity.z = lerp(this.velocity.z, desiredVz, 0.14 + feature.onset * 0.05);
    this.velocity.y -= this.point.y * 0.00034;
    this.velocity.z -= this.point.z * 0.00034;

    this.prevPoint.copy(this.point);
    this.point.add(this.velocity);
    this.tangent.subVectors(this.point, this.prevPoint).normalize();

    const radius = (Math.max(0.035, dx * 0.42) + feature.low * 0.075 + feature.energy * 0.038) * (0.88 + this.complexity * 0.14);
    const previousRadius = Math.max(0.026, radius * (0.92 + randAt(this.seed, stepIndex, 130) * 0.12));
    const trunkColor = this.colorFor(feature, progress, -0.10).clone();
    this.setCylinderInstance(this.trunk, this.trunkCount, this.prevPoint, this.point, previousRadius, radius, trunkColor);
    this.trunkCount++;
    this.trunk.count = this.trunkCount;
    this.trunk.instanceMatrix.needsUpdate = true;
    if (this.trunk.instanceColor) this.trunk.instanceColor.needsUpdate = true;
    this.expandBounds(this.point);

    if (feature.sectionIndex !== this.lastSection) {
      this.createSectionRing(feature, stepIndex);
      this.lastSection = feature.sectionIndex;
    }

    const branchSignal = feature.onset * 0.88 + feature.high * 0.16 + feature.mid * 0.04;
    const branchGap = Math.max(3, Math.round(8 - this.complexity * 3));
    const shouldBranch = stepIndex - this.lastBranchStep >= branchGap && branchSignal > 0.43 && randAt(this.seed, stepIndex, 140) < branchSignal * (0.34 + this.complexity * 0.10);
    if (shouldBranch && this.branchCount < this.branches.instanceMatrix.count) {
      const { basis1, basis2, endpoint } = this.tmp;
      const reference = Math.abs(this.tangent.y) > 0.85 ? new THREE.Vector3(0, 0, 1) : new THREE.Vector3(0, 1, 0);
      basis1.crossVectors(this.tangent, reference).normalize();
      basis2.crossVectors(this.tangent, basis1).normalize();
      const count = feature.high > 0.82 && feature.onset > 0.72 && this.complexity > 1.15 ? 2 : 1;
      for (let b = 0; b < count && this.branchCount < this.branches.instanceMatrix.count; b++) {
        const angle = randAt(this.seed, stepIndex, 142 + b) * TAU;
        const length = 0.42 + feature.onset * 1.65 + feature.high * 0.95 + randAt(this.seed, stepIndex, 146 + b) * 0.65;
        endpoint.copy(this.point)
          .addScaledVector(basis1, Math.cos(angle) * length)
          .addScaledVector(basis2, Math.sin(angle) * length)
          .addScaledVector(this.tangent, length * (0.08 + feature.mid * 0.18));
        const branchColor = this.colorFor(feature, progress, -0.01 + b * 0.035).clone();
        this.setCylinderInstance(this.branches, this.branchCount, this.point, endpoint, Math.max(0.018, radius * 0.42), Math.max(0.010, radius * 0.10), branchColor);
        this.branchCount++;
        this.expandBounds(endpoint);

        if (this.nodeCount < this.nodes.instanceMatrix.count) {
          const nodeScale = 0.055 + feature.onset * 0.16 + feature.mid * 0.075;
          this.setShapeInstance(this.nodes, this.nodeCount, endpoint, nodeScale, stepIndex * 3 + b, branchColor, 0.85 + randAt(this.seed, stepIndex, 149 + b) * 0.35);
          this.nodeCount++;
        }
      }
      this.lastBranchStep = stepIndex;
      this.branches.count = this.branchCount;
      this.branches.instanceMatrix.needsUpdate = true;
      if (this.branches.instanceColor) this.branches.instanceColor.needsUpdate = true;
      this.nodes.count = this.nodeCount;
      this.nodes.instanceMatrix.needsUpdate = true;
      if (this.nodes.instanceColor) this.nodes.instanceColor.needsUpdate = true;
    }

    if (feature.onset > 0.74 && stepIndex - this.lastNodeStep >= 10 && this.nodeCount < this.nodes.instanceMatrix.count && randAt(this.seed, stepIndex, 160) < feature.onset * 0.62) {
      const nodeColor = this.colorFor(feature, progress, 0.15).clone();
      const nodeScale = Math.max(0.07, radius * (1.35 + feature.onset * 1.15));
      this.setShapeInstance(this.nodes, this.nodeCount, this.point, nodeScale, stepIndex, nodeColor, 0.72 + feature.high * 0.5);
      this.nodeCount++;
      this.lastNodeStep = stepIndex;
      this.nodes.count = this.nodeCount;
      this.nodes.instanceMatrix.needsUpdate = true;
      if (this.nodes.instanceColor) this.nodes.instanceColor.needsUpdate = true;
    }

    if (feature.high > 0.67 && randAt(this.seed, stepIndex, 170) < feature.high * 0.24 * this.complexity) {
      const sparkN = 1 + Math.floor(feature.high * 2.2);
      const { basis1, basis2, endpoint } = this.tmp;
      const reference = Math.abs(this.tangent.z) > 0.85 ? new THREE.Vector3(0, 1, 0) : new THREE.Vector3(0, 0, 1);
      basis1.crossVectors(this.tangent, reference).normalize();
      basis2.crossVectors(this.tangent, basis1).normalize();
      for (let s = 0; s < sparkN && this.sparkCount < this.sparks.instanceMatrix.count; s++) {
        const angle = randAt(this.seed, stepIndex, 172 + s) * TAU;
        const distance = 0.18 + randAt(this.seed, stepIndex, 178 + s) * (0.55 + feature.high * 1.1);
        endpoint.copy(this.point)
          .addScaledVector(basis1, Math.cos(angle) * distance)
          .addScaledVector(basis2, Math.sin(angle) * distance)
          .addScaledVector(this.tangent, (randAt(this.seed, stepIndex, 184 + s) - 0.5) * 0.9);
        const sparkColor = this.colorFor(feature, progress, 0.19).clone();
        this.setShapeInstance(this.sparks, this.sparkCount, endpoint, 0.025 + feature.high * 0.070, stepIndex * 5 + s, sparkColor, 1.55);
        this.sparkCount++;
        this.expandBounds(endpoint);
      }
      this.sparks.count = this.sparkCount;
      this.sparks.instanceMatrix.needsUpdate = true;
      if (this.sparks.instanceColor) this.sparks.instanceColor.needsUpdate = true;
    }

    this.tip.position.copy(this.point);
    this.tip.scale.setScalar(0.65 + feature.onset * 1.25);
    this.tip.material.color.copy(this.colorFor(feature, progress, 0.24));
    this.tipLight.position.copy(this.point).add(new THREE.Vector3(0, 2, 3));
    this.tipLight.color.copy(this.tip.material.color);
  }

  setCameraMode(mode) {
    this.cameraMode = mode;
    if (mode !== 'manual') this.controls.enabled = true;
  }

  updateCamera(dt) {
    if (!this.hasBounds) {
      this.controls.update();
      return;
    }
    if (this.cameraMode === 'manual') {
      this.controls.update();
      return;
    }
    const center = this.bounds.getCenter(new THREE.Vector3());
    const size = this.bounds.getSize(new THREE.Vector3());
    const radius = Math.max(4.5, size.length() * 0.5);
    const fov = THREE.MathUtils.degToRad(this.camera.fov);
    const fitDistance = Math.max(15, radius / Math.sin(fov / 2) * 1.12);
    let desiredPosition;
    let desiredTarget;
    if (this.cameraMode === 'overview') {
      desiredPosition = center.clone().add(new THREE.Vector3(fitDistance * 0.72, fitDistance * 0.42, fitDistance * 0.86));
      desiredTarget = center;
    } else {
      // Ahead of the growth tangent, looking back through the current tip and the accumulated work.
      const ahead = this.tangent.clone().multiplyScalar(fitDistance * 0.72);
      const side = new THREE.Vector3(0, fitDistance * 0.19 + 2.5, fitDistance * 0.40 + 5);
      desiredPosition = this.point.clone().add(ahead).add(side);
      desiredTarget = center.clone().lerp(this.point, 0.48);
    }
    const smoothing = 1 - Math.exp(-dt * 2.1);
    this.camera.position.lerp(desiredPosition, smoothing);
    this.controls.target.lerp(desiredTarget, smoothing);
    this.camera.near = Math.max(0.03, fitDistance / 2500);
    this.camera.far = Math.max(1500, fitDistance * 12);
    this.camera.updateProjectionMatrix();
    this.controls.update();
  }

  render(dt) {
    this.updateCamera(dt);
    this.renderer.render(this.scene, this.camera);
  }

  exportPng() {
    this.renderer.render(this.scene, this.camera);
    return new Promise((resolve) => this.renderer.domElement.toBlob(resolve, 'image/png'));
  }
}

const els = {
  app: document.querySelector('#app'),
  stage: document.querySelector('#stage'),
  canvas2d: document.querySelector('#canvas2d'),
  threeContainer: document.querySelector('#threeContainer'),
  emptyState: document.querySelector('#emptyState'),
  dropZone: document.querySelector('#dropZone'),
  fileInput: document.querySelector('#fileInput'),
  openFileBtn: document.querySelector('#openFileBtn'),
  demoBtn: document.querySelector('#demoBtn'),
  modeButtons: Array.from(document.querySelectorAll('[data-mode]')),
  playBtn: document.querySelector('#playBtn'),
  playIcon: document.querySelector('#playIcon'),
  timeline: document.querySelector('#timeline'),
  currentTime: document.querySelector('#currentTime'),
  duration: document.querySelector('#duration'),
  trackTitle: document.querySelector('#trackTitle'),
  trackMeta: document.querySelector('#trackMeta'),
  transportTitle: document.querySelector('#transportTitle'),
  seedValue: document.querySelector('#seedValue'),
  regenerateBtn: document.querySelector('#regenerateBtn'),
  paletteSelect: document.querySelector('#paletteSelect'),
  paintingStyleSelect: document.querySelector('#paintingStyleSelect'),
  paintingStyleSetting: document.querySelector('#paintingStyleSetting'),
  paintingStyleDescription: document.querySelector('#paintingStyleDescription'),
  complexity: document.querySelector('#complexity'),
  complexityValue: document.querySelector('#complexityValue'),
  cameraBtn: document.querySelector('#cameraBtn'),
  cameraLabel: document.querySelector('#cameraLabel'),
  exportBtn: document.querySelector('#exportBtn'),
  replaceBtn: document.querySelector('#replaceBtn'),
  analysisOverlay: document.querySelector('#analysisOverlay'),
  analysisText: document.querySelector('#analysisText'),
  analysisProgress: document.querySelector('#analysisProgress'),
  sectionPill: document.querySelector('#sectionPill'),
  stylePill: document.querySelector('#stylePill'),
  energyBar: document.querySelector('#energyBar'),
  bassBar: document.querySelector('#bassBar'),
  trebleBar: document.querySelector('#trebleBar'),
  modeHint: document.querySelector('#modeHint'),
  mappingRules: document.querySelector('#mappingRules'),
  toast: document.querySelector('#toast'),
  loadingSpinner: document.querySelector('#loadingSpinner'),
};

const state = {
  audioContext: null,
  analyser: null,
  gain: null,
  buffer: null,
  analysis: null,
  source: null,
  sourceToken: 0,
  isPlaying: false,
  startedAt: 0,
  currentTime: 0,
  duration: 0,
  fileName: '',
  mode: '2d',
  seed: Math.floor(Math.random() * 0xffffffff) >>> 0,
  paletteKey: 'aurora',
  paintingStyleKey: 'painterly',
  complexity: 1,
  visualStep: 0.09,
  generatedStep: -1,
  scrubbing: false,
  lastFrameTime: performance.now(),
  loaded: false,
  rebuilding: false,
};

const painting = new CanvasPainting(els.canvas2d);

class SculptureFallback {
  constructor(container, error) {
    this.container = container;
    this.cameraMode = 'manual';
    this.tip = { visible: false };
    this.error = error;
    container.innerHTML = '<div class="webgl-fallback"><strong>当前环境无法启动 3D 模式</strong><span>2D 音乐绘画仍可正常使用。请在开启硬件加速的 Chrome、Edge、Safari 或 Firefox 中打开以体验 3D。</span></div>';
  }
  configure() {}
  resize() { return false; }
  reset() {}
  applyPalette() {}
  generate() {}
  render() {}
  setCameraMode() {}
  async exportPng() { throw new Error('当前浏览器未启用 WebGL，无法导出 3D 视图'); }
}

let sculpture;
let webglAvailable = true;
try {
  sculpture = new Sculpture3D(els.threeContainer);
} catch (error) {
  console.warn('3D mode unavailable:', error);
  webglAvailable = false;
  sculpture = new SculptureFallback(els.threeContainer, error);
}

function ensureAudioContext() {
  if (!state.audioContext) {
    const AudioContextClass = window.AudioContext || window.webkitAudioContext;
    if (!AudioContextClass) throw new Error('当前浏览器不支持 Web Audio API');
    state.audioContext = new AudioContextClass();
    state.analyser = state.audioContext.createAnalyser();
    state.analyser.fftSize = 2048;
    state.analyser.smoothingTimeConstant = 0.76;
    state.gain = state.audioContext.createGain();
    state.gain.gain.value = 0.92;
    state.analyser.connect(state.gain);
    state.gain.connect(state.audioContext.destination);
  }
  return state.audioContext;
}

function stopSource() {
  state.sourceToken++;
  if (state.source) {
    try { state.source.stop(); } catch (_) { /* already stopped */ }
    try { state.source.disconnect(); } catch (_) { /* ignore */ }
    state.source = null;
  }
}

async function play() {
  if (!state.buffer || !state.analysis) return;
  const ctx = ensureAudioContext();
  await ctx.resume();
  if (state.currentTime >= state.duration - 0.03) {
    setCurrentTime(0, true);
  }
  stopSource();
  const source = ctx.createBufferSource();
  const token = ++state.sourceToken;
  source.buffer = state.buffer;
  source.connect(state.analyser);
  source.onended = () => {
    if (token !== state.sourceToken || !state.isPlaying) return;
    state.currentTime = state.duration;
    state.isPlaying = false;
    updatePlayUI();
    sculpture.tip.visible = false;
    toast('作品已完成：所有痕迹已永久凝固');
  };
  state.startedAt = ctx.currentTime - state.currentTime;
  source.start(0, state.currentTime);
  state.source = source;
  state.isPlaying = true;
  sculpture.tip.visible = true;
  updatePlayUI();
}

function pause() {
  if (!state.isPlaying) return;
  state.currentTime = clamp(state.audioContext.currentTime - state.startedAt, 0, state.duration);
  state.isPlaying = false;
  stopSource();
  updatePlayUI();
}

function togglePlay() {
  if (state.isPlaying) pause();
  else play().catch((error) => showError(error));
}

function currentPlaybackTime() {
  if (state.isPlaying && state.audioContext) return clamp(state.audioContext.currentTime - state.startedAt, 0, state.duration);
  return state.currentTime;
}

function configureVisuals() {
  if (!state.analysis) return;
  state.visualStep = Math.max(0.075, state.duration / 9500);
  const config = {
    analysis: state.analysis,
    duration: state.duration,
    seed: state.seed,
    palette: PALETTES[state.paletteKey],
    complexity: state.complexity,
    paintingStyleKey: state.paintingStyleKey,
    stepDuration: state.visualStep,
  };
  painting.configure(config);
  sculpture.configure(config);
  applyPaletteToUI();
}

function clearVisuals() {
  painting.resize();
  painting.reset();
  sculpture.resize();
  sculpture.reset();
  state.generatedStep = -1;
}

function generateUntil(time) {
  if (!state.analysis || state.rebuilding) return;
  const target = Math.floor(clamp(time, 0, state.duration) / state.visualStep);
  while (state.generatedStep < target) {
    state.generatedStep++;
    const t = Math.min(state.duration, state.generatedStep * state.visualStep);
    const feature = sampleAnalysis(state.analysis, t);
    painting.generate(t, feature, state.generatedStep);
    sculpture.generate(t, feature, state.generatedStep);
  }
}

async function rebuildTo(time, showOverlay = false) {
  if (!state.analysis) return;
  state.rebuilding = true;
  if (showOverlay) showAnalysis('正在重建作品…', 0);
  clearVisuals();
  const target = Math.floor(clamp(time, 0, state.duration) / state.visualStep);
  for (let step = 0; step <= target; step++) {
    state.generatedStep = step;
    const t = Math.min(state.duration, step * state.visualStep);
    const feature = sampleAnalysis(state.analysis, t);
    painting.generate(t, feature, step);
    sculpture.generate(t, feature, step);
    if (showOverlay && step % 300 === 0) {
      updateAnalysisProgress(target > 0 ? step / target : 1, '正在重建作品…');
      await nextFrame();
    }
  }
  state.rebuilding = false;
  if (showOverlay) hideAnalysis();
}

function setCurrentTime(time, rebuild = false) {
  const wasPlaying = state.isPlaying;
  if (wasPlaying) pause();
  state.currentTime = clamp(time, 0, state.duration);
  if (rebuild) rebuildTo(state.currentTime, state.duration > 900).catch(showError);
  updateTimelineUI();
  if (wasPlaying) play().catch(showError);
}

function updatePlayUI() {
  els.playBtn.disabled = !state.loaded;
  els.playBtn.setAttribute('aria-label', state.isPlaying ? '暂停' : '播放');
  els.playIcon.innerHTML = state.isPlaying
    ? '<rect x="7" y="5" width="3.5" height="14" rx="1"></rect><rect x="13.5" y="5" width="3.5" height="14" rx="1"></rect>'
    : '<path d="M8 5.5v13l10-6.5z"></path>';
}

function updateTimelineUI(time = currentPlaybackTime()) {
  const p = state.duration ? clamp(time / state.duration) : 0;
  if (!state.scrubbing) els.timeline.value = String(p * 1000);
  els.timeline.style.setProperty('--progress', `${p * 100}%`);
  els.currentTime.textContent = formatTime(time);
  els.duration.textContent = formatTime(state.duration);
}

function updateFeatureUI(time) {
  if (!state.analysis) return;
  const f = sampleAnalysis(state.analysis, time);
  els.energyBar.style.transform = `scaleX(${clamp(f.energy)})`;
  els.bassBar.style.transform = `scaleX(${clamp(f.low)})`;
  els.trebleBar.style.transform = `scaleX(${clamp(f.high)})`;
  els.sectionPill.textContent = `段落 ${f.sectionIndex + 1} / ${state.analysis.sections.length}`;
}

function updatePaintingStyleUI() {
  const style = PAINTING_STYLES[state.paintingStyleKey] || PAINTING_STYLES.trace;
  if (els.paintingStyleSelect) els.paintingStyleSelect.value = state.paintingStyleKey;
  if (els.paintingStyleDescription) els.paintingStyleDescription.textContent = style.description;
  els.stage.dataset.paintingStyle = state.paintingStyleKey;
}

function updateModeUI() {
  if (state.mode === '3d' && !webglAvailable) {
    state.mode = '2d';
    toast('当前环境未启用 WebGL，已保留 2D 模式');
  }
  for (const button of els.modeButtons) {
    const active = button.dataset.mode === state.mode;
    button.classList.toggle('active', active);
    button.setAttribute('aria-pressed', String(active));
  }
  els.stage.dataset.mode = state.mode;
  const paintingStyle = PAINTING_STYLES[state.paintingStyleKey] || PAINTING_STYLES.trace;
  if (state.mode === '2d') {
    els.modeHint.textContent = paintingStyle.hint;
    els.stylePill.textContent = paintingStyle.label;
    els.mappingRules.innerHTML = paintingStyle.mappingHtml;
    els.paintingStyleSetting.hidden = false;
  } else {
    els.modeHint.textContent = '结构沿时间连续生长，生成后不移动、不消失';
    els.stylePill.textContent = '3D · 音乐建筑';
    els.mappingRules.innerHTML = '时间 → 生长距离<br>低频 → 主体与地基<br>中频 → 空间路径<br>高频 → 分支与晶体<br>瞬态 → 节点与结构重音<br>段落 → 建筑转折与空间环';
    els.paintingStyleSetting.hidden = true;
  }
  els.cameraBtn.hidden = state.mode !== '3d' || !webglAvailable;
  els.exportBtn.querySelector('span').textContent = state.mode === '2d' ? '导出画作' : '导出视图';
  updatePaintingStyleUI();
  if (state.mode === '3d') sculpture.resize();
}
function updateSeedUI() {
  els.seedValue.textContent = state.seed.toString(16).toUpperCase().padStart(8, '0');
}

function updateCameraModeUI() {
  if (!els.cameraLabel) return;
  const labels = { follow: '前沿跟随', overview: '整体总览', manual: '手动视角' };
  els.cameraLabel.textContent = labels[sculpture.cameraMode] || '前沿跟随';
  els.cameraBtn.dataset.cameraMode = sculpture.cameraMode;
}

function applyPaletteToUI() {
  const p = PALETTES[state.paletteKey];
  document.documentElement.style.setProperty('--palette-accent', p.accent);
  document.documentElement.style.setProperty('--stage-bg', p.bg);
  els.app.dataset.lightPalette = String(!p.dark);
}

function showAnalysis(text, progress = 0) {
  els.analysisText.textContent = text;
  els.analysisProgress.style.transform = `scaleX(${clamp(progress)})`;
  els.analysisOverlay.hidden = false;
}

function updateAnalysisProgress(progress, text = null) {
  if (text) els.analysisText.textContent = text;
  els.analysisProgress.style.transform = `scaleX(${clamp(progress)})`;
}

function hideAnalysis() {
  els.analysisOverlay.hidden = true;
}

let toastTimer = null;
function toast(message) {
  clearTimeout(toastTimer);
  els.toast.textContent = message;
  els.toast.classList.add('show');
  toastTimer = setTimeout(() => els.toast.classList.remove('show'), 2600);
}

function showError(error) {
  console.error(error);
  hideAnalysis();
  toast(error?.message || '发生错误，请换一个音频文件重试');
}

async function loadAudioBuffer(buffer, name, meta = '') {
  pause();
  state.loaded = false;
  state.buffer = buffer;
  state.duration = buffer.duration;
  state.currentTime = 0;
  state.fileName = name;
  els.trackTitle.textContent = name.replace(/\.[^/.]+$/, '');
  els.transportTitle.textContent = name.replace(/\.[^/.]+$/, '');
  els.trackMeta.textContent = meta || `${buffer.numberOfChannels === 1 ? '单声道' : '立体声'} · ${(buffer.sampleRate / 1000).toFixed(1)} kHz`;
  els.duration.textContent = formatTime(buffer.duration);
  els.emptyState.hidden = true;
  els.timeline.disabled = true;
  els.exportBtn.disabled = true;
  els.regenerateBtn.disabled = true;
  showAnalysis('正在理解歌曲结构…', 0.02);
  state.analysis = await analyzeAudioBuffer(buffer, (p) => updateAnalysisProgress(p, p < 0.52 ? '正在辨认低频、旋律与瞬态…' : '正在寻找段落与构图骨架…'));
  configureVisuals();
  clearVisuals();
  state.loaded = true;
  els.timeline.disabled = false;
  els.exportBtn.disabled = false;
  els.regenerateBtn.disabled = false;
  hideAnalysis();
  updatePlayUI();
  updateTimelineUI(0);
  updateSeedUI();
  updateFeatureUI(0);
  toast(`分析完成：识别出 ${state.analysis.sections.length} 个音乐段落`);
}

async function loadFile(file) {
  if (!file) return;
  if (!file.type.startsWith('audio/') && !/\.(mp3|wav|m4a|aac|ogg|flac)$/i.test(file.name)) {
    throw new Error('请选择 MP3、WAV、M4A、AAC、OGG 或 FLAC 音频文件');
  }
  const ctx = ensureAudioContext();
  showAnalysis('正在读取音频…', 0.01);
  const arrayBuffer = await file.arrayBuffer();
  updateAnalysisProgress(0.03, '正在解码音频…');
  const decoded = await ctx.decodeAudioData(arrayBuffer.slice(0));
  const sizeMb = (file.size / 1024 / 1024).toFixed(1);
  await loadAudioBuffer(decoded, file.name, `${sizeMb} MB · ${decoded.numberOfChannels === 1 ? '单声道' : '立体声'} · ${(decoded.sampleRate / 1000).toFixed(1)} kHz`);
}

function addSine(bufferData, sampleRate, start, duration, frequency, amplitude, decay = 0, phase = 0) {
  const startIndex = Math.floor(start * sampleRate);
  const endIndex = Math.min(bufferData.length, Math.floor((start + duration) * sampleRate));
  for (let i = startIndex; i < endIndex; i++) {
    const t = (i - startIndex) / sampleRate;
    const env = decay > 0 ? Math.exp(-decay * t) : Math.sin(Math.PI * clamp(t / duration));
    bufferData[i] += Math.sin(TAU * frequency * t + phase) * amplitude * env;
  }
}

function addNoise(bufferData, sampleRate, start, duration, amplitude, seed) {
  const startIndex = Math.floor(start * sampleRate);
  const endIndex = Math.min(bufferData.length, Math.floor((start + duration) * sampleRate));
  let last = 0;
  for (let i = startIndex; i < endIndex; i++) {
    const t = (i - startIndex) / sampleRate;
    const env = Math.exp(-18 * t);
    const noise = randAt(seed, i, 500) * 2 - 1;
    last = last * 0.35 + noise * 0.65;
    bufferData[i] += last * amplitude * env;
  }
}

async function createDemoBuffer() {
  const ctx = ensureAudioContext();
  const sampleRate = 44100;
  const duration = 24;
  const buffer = ctx.createBuffer(2, duration * sampleRate, sampleRate);
  const left = buffer.getChannelData(0);
  const right = buffer.getChannelData(1);
  const bpm = 108;
  const beat = 60 / bpm;
  const bassNotes = [65.41, 73.42, 87.31, 98.00, 65.41, 82.41, 73.42, 98.00];
  const melody = [261.63, 293.66, 329.63, 392.00, 440.00, 392.00, 329.63, 293.66, 349.23, 392.00, 523.25, 440.00, 392.00, 329.63, 293.66, 261.63];

  for (let t = 0, beatIndex = 0; t < duration; t += beat, beatIndex++) {
    addSine(left, sampleRate, t, 0.35, 52, 0.34, 12);
    addSine(right, sampleRate, t, 0.35, 52, 0.31, 12, 0.04);
    if (beatIndex % 2 === 1) {
      addNoise(left, sampleRate, t, 0.16, 0.16, 770 + beatIndex);
      addNoise(right, sampleRate, t, 0.16, 0.18, 990 + beatIndex);
      addSine(left, sampleRate, t, 0.14, 180, 0.07, 20);
      addSine(right, sampleRate, t, 0.14, 190, 0.06, 20);
    }
    for (let h = 0; h < 2; h++) {
      const ht = t + h * beat * 0.5;
      addNoise(left, sampleRate, ht, 0.06, 0.045 + (beatIndex % 4 === 3 ? 0.025 : 0), 3300 + beatIndex * 3 + h);
      addNoise(right, sampleRate, ht, 0.06, 0.05, 4400 + beatIndex * 3 + h);
    }
    const bass = bassNotes[Math.floor(beatIndex / 2) % bassNotes.length];
    addSine(left, sampleRate, t, beat * 0.92, bass, 0.10, 0);
    addSine(right, sampleRate, t, beat * 0.92, bass, 0.095, 0, 0.12);
    addSine(left, sampleRate, t, beat * 0.92, bass * 2, 0.028, 0);
    addSine(right, sampleRate, t, beat * 0.92, bass * 2, 0.026, 0, 0.08);
  }

  const chordRoots = [130.81, 146.83, 174.61, 196.00];
  for (let section = 0; section < 4; section++) {
    const start = section * 6;
    const root = chordRoots[section];
    const chord = [root, root * Math.pow(2, 4 / 12), root * Math.pow(2, 7 / 12)];
    for (const f of chord) {
      addSine(left, sampleRate, start, 6, f, 0.032 + section * 0.006, 0);
      addSine(right, sampleRate, start, 6, f * 1.0018, 0.030 + section * 0.006, 0, 0.18);
      addSine(left, sampleRate, start, 6, f * 2, 0.012, 0);
      addSine(right, sampleRate, start, 6, f * 2.002, 0.011, 0, 0.1);
    }
  }

  for (let i = 0; i < melody.length * 2; i++) {
    const t = 5.5 + i * beat * 0.5;
    if (t >= duration - 0.2) break;
    const f = melody[i % melody.length] * (i > melody.length ? 1.0 : 0.5);
    const amp = i > 15 ? 0.065 : 0.045;
    addSine(left, sampleRate, t, beat * 0.46, f, amp, 3.2);
    addSine(right, sampleRate, t + 0.018, beat * 0.46, f * 1.002, amp * 0.94, 3.2, 0.2);
    addSine(left, sampleRate, t, beat * 0.42, f * 2, amp * 0.22, 4.1);
  }

  let peak = 0;
  for (let i = 0; i < left.length; i++) peak = Math.max(peak, Math.abs(left[i]), Math.abs(right[i]));
  const gain = peak > 0 ? 0.88 / peak : 1;
  for (let i = 0; i < left.length; i++) { left[i] *= gain; right[i] *= gain; }
  return buffer;
}

async function loadDemo(autoPlay = false, previewProgress = 0) {
  showAnalysis('正在生成内置示例音乐…', 0.01);
  const buffer = await createDemoBuffer();
  await loadAudioBuffer(buffer, '内置示例：光的建筑', '24 秒 · 合成鼓、贝斯、和弦与旋律');
  if (previewProgress > 0) {
    state.currentTime = state.duration * previewProgress;
    await rebuildTo(state.currentTime, false);
    updateTimelineUI();
  }
  if (autoPlay) await play();
}

async function regenerate() {
  if (!state.analysis) return;
  const wasPlaying = state.isPlaying;
  if (wasPlaying) pause();
  state.seed = Math.floor(Math.random() * 0xffffffff) >>> 0;
  updateSeedUI();
  configureVisuals();
  await rebuildTo(state.currentTime, state.duration > 600);
  if (wasPlaying) await play();
  toast('已更换生成种子：歌曲结构保留，局部形态重新创作');
}

function downloadBlob(blob, filename) {
  if (!blob) throw new Error('当前浏览器无法导出图像');
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

async function exportArtwork() {
  if (!state.loaded) return;
  const baseName = (state.fileName || '音乐画布').replace(/\.[^/.]+$/, '').replace(/[\\/:*?"<>|]/g, '_');
  if (state.mode === '2d') {
    const blob = await new Promise((resolve) => els.canvas2d.toBlob(resolve, 'image/png'));
    const style = PAINTING_STYLES[state.paintingStyleKey] || PAINTING_STYLES.trace;
    downloadBlob(blob, `${baseName}_2D_${style.exportName}_${els.seedValue.textContent}.png`);
  } else {
    const blob = await sculpture.exportPng();
    downloadBlob(blob, `${baseName}_3D_${els.seedValue.textContent}.png`);
  }
  toast('已导出当前作品视图');
}

function cycleCameraMode() {
  const next = sculpture.cameraMode === 'follow' ? 'overview' : 'follow';
  sculpture.setCameraMode(next);
  updateCameraModeUI();
}

function bindEvents() {
  els.openFileBtn.addEventListener('click', () => els.fileInput.click());
  els.replaceBtn.addEventListener('click', () => els.fileInput.click());
  els.fileInput.addEventListener('change', () => loadFile(els.fileInput.files[0]).catch(showError));
  els.demoBtn.addEventListener('click', () => loadDemo(true).catch(showError));
  els.playBtn.addEventListener('click', togglePlay);
  els.regenerateBtn.addEventListener('click', () => regenerate().catch(showError));
  els.exportBtn.addEventListener('click', () => exportArtwork().catch(showError));
  els.cameraBtn.addEventListener('click', cycleCameraMode);

  for (const button of els.modeButtons) {
    button.addEventListener('click', () => {
      state.mode = button.dataset.mode;
      updateModeUI();
    });
  }

  els.paintingStyleSelect.addEventListener('change', async () => {
    const nextStyle = els.paintingStyleSelect.value;
    if (!PAINTING_STYLES[nextStyle] || nextStyle === state.paintingStyleKey) return;
    const wasPlaying = state.isPlaying;
    if (wasPlaying) pause();
    state.paintingStyleKey = nextStyle;
    updateModeUI();
    if (state.analysis) {
      configureVisuals();
      await rebuildTo(state.currentTime, state.duration > 600);
    } else {
      painting.styleKey = nextStyle;
      painting.reset();
    }
    if (wasPlaying) await play();
    toast(`已切换到${PAINTING_STYLES[nextStyle].label}`);
  });

  els.paletteSelect.addEventListener('change', async () => {
    if (!PALETTES[els.paletteSelect.value]) return;
    const wasPlaying = state.isPlaying;
    if (wasPlaying) pause();
    state.paletteKey = els.paletteSelect.value;
    configureVisuals();
    await rebuildTo(state.currentTime, state.duration > 600);
    if (wasPlaying) await play();
  });

  els.complexity.addEventListener('input', () => {
    state.complexity = Number(els.complexity.value);
    els.complexityValue.textContent = state.complexity.toFixed(1);
  });
  els.complexity.addEventListener('change', async () => {
    if (!state.analysis) return;
    const wasPlaying = state.isPlaying;
    if (wasPlaying) pause();
    configureVisuals();
    await rebuildTo(state.currentTime, state.duration > 600);
    if (wasPlaying) await play();
  });

  els.timeline.addEventListener('pointerdown', () => { state.scrubbing = true; });
  els.timeline.addEventListener('input', () => {
    const t = Number(els.timeline.value) / 1000 * state.duration;
    els.currentTime.textContent = formatTime(t);
    els.timeline.style.setProperty('--progress', `${Number(els.timeline.value) / 10}%`);
  });
  const finishScrub = () => {
    if (!state.scrubbing) return;
    state.scrubbing = false;
    const t = Number(els.timeline.value) / 1000 * state.duration;
    setCurrentTime(t, true);
  };
  els.timeline.addEventListener('change', finishScrub);
  els.timeline.addEventListener('pointerup', finishScrub);

  const prevent = (event) => { event.preventDefault(); event.stopPropagation(); };
  ['dragenter', 'dragover'].forEach((type) => document.addEventListener(type, (event) => {
    prevent(event);
    els.dropZone.classList.add('dragging');
  }));
  ['dragleave', 'drop'].forEach((type) => document.addEventListener(type, (event) => {
    prevent(event);
    els.dropZone.classList.remove('dragging');
  }));
  document.addEventListener('drop', (event) => {
    const file = event.dataTransfer?.files?.[0];
    if (file) loadFile(file).catch(showError);
  });

  window.addEventListener('resize', () => {
    const changed2d = painting.resize();
    sculpture.resize();
    if (changed2d && state.analysis) rebuildTo(state.currentTime, false).catch(showError);
  });

  window.addEventListener('keydown', (event) => {
    if (event.target.matches('input, select, button')) return;
    if (event.code === 'Space') { event.preventDefault(); togglePlay(); }
    if (event.key.toLowerCase() === 'm') {
      state.mode = state.mode === '2d' ? '3d' : '2d';
      updateModeUI();
    }
  });
}

function animate(now) {
  requestAnimationFrame(animate);
  const dt = Math.min(0.1, Math.max(0.001, (now - state.lastFrameTime) / 1000));
  state.lastFrameTime = now;
  const time = currentPlaybackTime();
  if (state.isPlaying) {
    state.currentTime = time;
    generateUntil(time);
    if (time >= state.duration - 0.015) {
      state.currentTime = state.duration;
      state.isPlaying = false;
      stopSource();
      updatePlayUI();
      sculpture.tip.visible = false;
    }
  }
  updateTimelineUI(time);
  updateFeatureUI(time);
  sculpture.resize();
  sculpture.render(dt);
}

async function init() {
  bindEvents();
  if (!webglAvailable) {
    const button3d = els.modeButtons.find((button) => button.dataset.mode === '3d');
    if (button3d) { button3d.disabled = true; button3d.title = '当前环境未启用 WebGL'; }
  }
  updateSeedUI();
  updatePlayUI();
  updatePaintingStyleUI();
  updateModeUI();
  updateCameraModeUI();
  painting.resize();
  painting.palette = PALETTES[state.paletteKey];
  painting.reset();
  sculpture.resize();
  sculpture.applyPalette();
  requestAnimationFrame(animate);

  const params = new URLSearchParams(location.search);
  if (params.has('preview')) {
    const mode = params.get('preview') === '2d' ? '2d' : '3d';
    state.mode = mode;
    updateModeUI();
    await loadDemo(false, mode === '2d' ? 0.88 : 0.72);
  }
}

init().catch(showError);

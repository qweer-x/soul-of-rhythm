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
    description: '完整保留最初的生成方式：低频色带、旋律主线、高频细线与瞬态叶片沿时间连续生长，更接近声音轨迹与生成图案。',
    hint: '风格 1 · 时间从左向右，声音凝固为流线、色带与叶片',
    mappingHtml: '时间 → 横向轨迹<br>低频 → 连续色带<br>中频 → 旋律主线<br>高频 → 细线与光点<br>瞬态 → 叶片与节点<br>段落 → 色场转折',
    exportName: '风格1_声纹流线',
  },
  painterly: {
    order: 2,
    name: '声景画卷',
    label: '风格 2 · 声景画卷',
    description: '把歌曲组织成一幅横向展开的抽象风景画：段落成为远山、树林、水面或亭廊，低中高频分别沉积为地景、主体与天空细节。',
    hint: '风格 2 · 每个段落成为一幕风景，音乐从左向右完成整幅画卷',
    mappingHtml: '时间 → 横向画卷逐步展开<br>低频 → 地形、山体与前景厚度<br>中频 / 音高 → 树木、建筑与主体位置<br>高频 → 云气、飞鸟与天空纹理<br>瞬态 → 山峰、树干、门廊与书写性重笔<br>段落 → 一幕独立风景与视觉焦点',
    exportName: '风格2_声景画卷',
  },
};

const SCULPTURE_STYLES = {
  growth: {
    order: 1,
    name: '生长脉络',
    label: '风格 1 · 生长脉络',
    description: '完整保留原有 3D 生成方式：音乐沿一条空间脉络连续生长，分支、晶体、节点和段落环生成后永久凝固。',
    hint: '风格 1 · 音乐沿空间脉络连续生长，所有结构永久凝固',
    mappingHtml: '时间 → 生长距离<br>低频 → 主干厚度<br>中频 → 空间路径<br>高频 → 分支与晶体<br>瞬态 → 节点与结构重音<br>段落 → 转折与空间环',
    exportName: '风格1_生长脉络',
  },
  sanctuary: {
    order: 2,
    name: '共鸣殿堂',
    label: '风格 2 · 共鸣殿堂',
    description: '把歌曲建造成一条可进入的音乐殿堂：低频铺设地基，中频抬起柱廊与穹顶，高频生成光片与悬挂晶体，段落成为连续厅室。',
    hint: '风格 2 · 音乐逐步建成地基、柱廊、门庭与光的穹顶',
    mappingHtml: '时间 → 建筑轴线与厅室深度<br>低频 → 地基、台阶与殿堂宽度<br>中频 / 音高 → 柱高、穹顶与空间比例<br>高频 → 光片、悬晶与细部<br>瞬态 → 门庭、横梁、祭台与结构重音<br>段落 → 一座新厅室与入口门廊',
    exportName: '风格2_共鸣殿堂',
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
    this.displayCtx = canvas.getContext('2d', { alpha: false });
    this.ctx = this.displayCtx;
    this.baseCanvas = document.createElement('canvas');
    this.terrainCanvas = document.createElement('canvas');
    this.detailCanvas = document.createElement('canvas');
    this.baseCtx = this.baseCanvas.getContext('2d', { alpha: false });
    this.terrainCtx = this.terrainCanvas.getContext('2d');
    this.detailCtx = this.detailCanvas.getContext('2d');
    this.width = 0;
    this.height = 0;
    this.dpr = 1;
    this.prev = null;
    this.painterPrev = null;
    this.painterGroundPrev = null;
    this.painterLastSubjectX = -Infinity;
    this.painterLastCloudX = -Infinity;
    this.painterLastAccentX = -Infinity;
    this.painterLastBirdX = -Infinity;
    this.lastSection = -1;
    this.palette = PALETTES.aurora;
    this.seed = 1;
    this.complexity = 1;
    this.duration = 1;
    this.analysis = null;
    this.styleKey = 'painterly';
    this.painterLayouts = [];
    this.painterTerrainPoints = [];
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
    const pixelWidth = Math.floor(width * dpr);
    const pixelHeight = Math.floor(height * dpr);
    this.canvas.width = pixelWidth;
    this.canvas.height = pixelHeight;
    for (const layer of [this.baseCanvas, this.terrainCanvas, this.detailCanvas]) {
      layer.width = pixelWidth;
      layer.height = pixelHeight;
    }
    for (const context of [this.displayCtx, this.baseCtx, this.terrainCtx, this.detailCtx]) {
      context.setTransform(dpr, 0, 0, dpr, 0, 0);
    }
    this.ctx = this.styleKey === 'painterly' ? this.detailCtx : this.displayCtx;
    return true;
  }

  reset() {
    if (this.styleKey === 'painterly') this.resetPainterly();
    else this.resetTrace();
  }

  resetTrace() {
    this.ctx = this.displayCtx;
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
    this.painterLastSubjectX = -Infinity;
    this.painterLastCloudX = -Infinity;
    this.painterLastAccentX = -Infinity;
    this.painterLastBirdX = -Infinity;
    this.painterTerrainPoints = [];
    this.lastSection = -1;
  }

  resetPainterly() {
    const { width: w, height: h, palette } = this;
    const baseFeature = { energy: 0.32, low: 0.40, mid: 0.36, high: 0.22, centroid: 0.34 };
    for (const context of [this.terrainCtx, this.detailCtx]) {
      context.save();
      context.setTransform(1, 0, 0, 1, 0, 0);
      context.clearRect(0, 0, this.canvas.width, this.canvas.height);
      context.restore();
      context.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    }
    this.ctx = this.baseCtx;
    const ctx = this.ctx;
    ctx.save();
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);

    const base = ctx.createLinearGradient(0, 0, 0, h);
    base.addColorStop(0, palette.dark ? palette.bg2 : palette.bg);
    base.addColorStop(0.58, palette.bg);
    base.addColorStop(1, palette.dark ? '#05070b' : palette.bg2);
    ctx.fillStyle = base;
    ctx.fillRect(0, 0, w, h);

    // A single calm underpainting anchors the composition. It is laid down once,
    // so the evolving song never turns into a stack of vertical spectrum bars.
    const groundWash = ctx.createLinearGradient(0, h * 0.48, 0, h);
    groundWash.addColorStop(0, this.paintColor(baseFeature, 0.18, 0, 0, 0.04, 0.48));
    groundWash.addColorStop(0.28, this.paintColor(baseFeature, 0.34, 0, palette.dark ? 0.055 : 0.035, -0.10, 0.52));
    groundWash.addColorStop(1, this.paintColor(baseFeature, 0.76, 0, palette.dark ? 0.24 : 0.14, -0.18, 0.46));
    ctx.fillStyle = groundWash;
    ctx.fillRect(0, h * 0.46, w, h * 0.54);

    ctx.globalCompositeOperation = 'source-over';
    for (let i = 0; i < 5; i++) {
      const y = h * (0.16 + i * 0.16 + (randAt(this.seed, i, 610) - 0.5) * 0.035);
      const gradient = ctx.createLinearGradient(0, y, w, y + h * 0.05);
      gradient.addColorStop(0, this.paintColor(baseFeature, 0, i, 0));
      gradient.addColorStop(0.28, this.paintColor(baseFeature, 0.28, i, palette.dark ? 0.045 : 0.030, -0.08 + i * 0.018));
      gradient.addColorStop(0.72, this.paintColor(baseFeature, 0.72, i, palette.dark ? 0.035 : 0.024, 0.03));
      gradient.addColorStop(1, this.paintColor(baseFeature, 1, i, 0));
      ctx.fillStyle = gradient;
      ctx.fillRect(0, y - h * 0.05, w, h * 0.12);
    }

    const fibreCount = Math.min(680, Math.max(230, Math.round((w * h) / 2500)));
    ctx.lineCap = 'round';
    for (let i = 0; i < fibreCount; i++) {
      const x = randAt(this.seed, i, 620) * w;
      const y = randAt(this.seed, i, 621) * h;
      const length = 2 + randAt(this.seed, i, 622) * 18;
      const angle = (randAt(this.seed, i, 623) - 0.5) * 0.26;
      ctx.beginPath();
      ctx.moveTo(x, y);
      ctx.lineTo(x + Math.cos(angle) * length, y + Math.sin(angle) * length);
      ctx.lineWidth = 0.25 + randAt(this.seed, i, 624) * 0.48;
      ctx.strokeStyle = palette.dark
        ? `rgba(255,255,255,${0.009 + randAt(this.seed, i, 625) * 0.015})`
        : `rgba(30,36,33,${0.012 + randAt(this.seed, i, 625) * 0.018})`;
      ctx.stroke();
    }

    const vignette = ctx.createRadialGradient(w * 0.52, h * 0.45, Math.min(w, h) * 0.16, w * 0.52, h * 0.45, Math.max(w, h) * 0.74);
    vignette.addColorStop(0, 'rgba(0,0,0,0)');
    vignette.addColorStop(1, palette.dark ? 'rgba(0,0,0,0.20)' : 'rgba(35,42,39,0.075)');
    ctx.fillStyle = vignette;
    ctx.fillRect(0, 0, w, h);
    ctx.restore();

    for (const layout of this.painterLayouts) layout.landmarkDrawn = false;
    this.prev = null;
    this.painterPrev = null;
    this.painterGroundPrev = null;
    this.painterLastSubjectX = -Infinity;
    this.painterLastCloudX = -Infinity;
    this.painterLastAccentX = -Infinity;
    this.painterLastBirdX = -Infinity;
    this.painterTerrainPoints = [];
    this.lastSection = -1;
    this.ctx = this.detailCtx;
    this.composePainterly();
  }

  composePainterly() {
    const ctx = this.displayCtx;
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
    ctx.drawImage(this.baseCanvas, 0, 0);
    ctx.drawImage(this.terrainCanvas, 0, 0);
    ctx.drawImage(this.detailCanvas, 0, 0);
    ctx.restore();
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    this.ctx = this.detailCtx;
  }

  color(feature, progress, sectionIndex, alpha = 1, lightShift = 0) {
    const { hue, saturation, lightness } = paletteHsl(this.palette, feature, progress, sectionIndex, this.seed, lightShift);
    return `hsla(${Math.round(hue * 360)}, ${Math.round(saturation * 100)}%, ${Math.round(lightness * 100)}%, ${alpha})`;
  }

  buildPainterlyLayouts() {
    this.painterLayouts = [];
    if (!this.analysis?.sections?.length) return;
    let horizon = clamp(0.57 + (randAt(this.seed, 0, 680) - 0.5) * 0.09, 0.49, 0.66);
    let ground = clamp(0.79 + (randAt(this.seed, 0, 681) - 0.5) * 0.05, 0.72, 0.84);

    for (let i = 0; i < this.analysis.sections.length; i++) {
      const section = this.analysis.sections[i];
      const a = section.averages;
      const start = clamp(section.start / this.duration);
      const end = clamp(section.end / this.duration);
      const span = Math.max(0.035, end - start);
      const horizonTarget = clamp(horizon + (a.high - a.low) * 0.055 + (randAt(this.seed, i, 682) - 0.5) * 0.075, 0.45, 0.68);
      const groundTarget = clamp(ground + (a.low - 0.5) * 0.035 + (randAt(this.seed, i, 683) - 0.5) * 0.035, 0.70, 0.86);

      let motif;
      const roll = randAt(this.seed, i, 684);
      if (a.low > a.high + 0.17) motif = 0;
      else if (a.high > a.low + 0.18) motif = 2;
      else if (a.mid > 0.58 && roll > 0.42) motif = 1;
      else motif = roll < 0.30 ? 0 : roll < 0.58 ? 1 : roll < 0.80 ? 2 : 3;
      const previousMotif = this.painterLayouts[i - 1]?.motif;
      if (previousMotif === motif) motif = (motif + 1 + Math.floor(randAt(this.seed, i, 688) * 2)) % 4;

      const focusProgress = clamp(start + span * (0.30 + randAt(this.seed, i, 685) * 0.42), start + span * 0.18, end - span * 0.10);
      this.painterLayouts.push({
        start,
        end,
        horizonStart: horizon,
        horizonEnd: horizonTarget,
        groundStart: ground,
        groundEnd: groundTarget,
        focusProgress,
        motif,
        phase: randAt(this.seed, i, 686) * TAU,
        ridgeScale: 0.78 + a.energy * 0.48 + randAt(this.seed, i, 687) * 0.20,
        subjectScale: 0.78 + a.mid * 0.40 + a.energy * 0.22,
        openness: clamp(0.35 + a.high * 0.48 - a.low * 0.16, 0.22, 0.84),
        landmarkDrawn: false,
      });
      horizon = horizonTarget;
      ground = groundTarget;
    }
  }

  paintColor(feature, progress, sectionIndex, alpha = 1, lightShift = 0, saturationScale = 0.82) {
    const { hue, saturation, lightness } = paletteHsl(this.palette, feature, progress, sectionIndex, this.seed, lightShift);
    const sat = clamp(saturation * saturationScale, 0.18, 0.78);
    const lit = this.palette.dark ? clamp(lightness * 0.88 + 0.035, 0.18, 0.74) : clamp(lightness * 0.78 + 0.06, 0.16, 0.70);
    return `hsla(${Math.round(hue * 360)}, ${Math.round(sat * 100)}%, ${Math.round(lit * 100)}%, ${alpha})`;
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

  painterlyMetrics(feature, progress) {
    const h = this.height;
    const layout = this.painterLayouts[feature.sectionIndex] || {
      start: 0, end: 1, horizonStart: 0.56, horizonEnd: 0.58,
      groundStart: 0.78, groundEnd: 0.79, focusProgress: 0.5,
      motif: 0, phase: 0, ridgeScale: 1, subjectScale: 1, openness: 0.5,
      landmarkDrawn: false,
    };
    const local = feature.sectionPhase;
    const horizon = lerp(layout.horizonStart, layout.horizonEnd, smoothstep(0, 1, local));
    const ground = lerp(layout.groundStart, layout.groundEnd, smoothstep(0, 1, local));
    const pitch = feature.dominant > 1
      ? clamp((Math.log2(Math.max(55, feature.dominant)) - Math.log2(55)) / 6)
      : feature.centroid;

    // Large-scale noise changes slowly. Musical detail is expressed by brush texture,
    // not by making the skyline jump on every analysis frame.
    const farNoise = valueNoise1D(progress * (2.2 + this.complexity * 0.55), this.seed, 900 + feature.sectionIndex) - 0.5;
    const midNoise = valueNoise1D(progress * (3.8 + this.complexity * 0.80), this.seed, 930 + feature.sectionIndex) - 0.5;
    const groundNoise = valueNoise1D(progress * (5.2 + this.complexity * 0.95), this.seed, 960 + feature.sectionIndex) - 0.5;
    const farY = h * clamp(horizon - 0.125 - feature.energy * 0.030 - farNoise * 0.090 * layout.ridgeScale, 0.20, 0.62);
    const midY = h * clamp(horizon - 0.015 - feature.low * 0.048 - midNoise * 0.065 * layout.ridgeScale, 0.34, 0.72);
    const groundY = h * clamp(ground - feature.low * 0.020 - groundNoise * 0.020, 0.64, 0.88);
    const subjectY = h * clamp(horizon - (pitch - 0.5) * 0.20 + midNoise * 0.025, 0.18, 0.74);
    const waterY = h * clamp(horizon + 0.060 + groundNoise * 0.016, 0.48, 0.74);

    const section = this.analysis?.sections?.[feature.sectionIndex];
    const averages = section?.averages || feature;
    const paintFeature = {
      energy: lerp(averages.energy ?? feature.energy, feature.energy, 0.08),
      low: lerp(averages.low ?? feature.low, feature.low, 0.07),
      mid: lerp(averages.mid ?? feature.mid, feature.mid, 0.07),
      high: lerp(averages.high ?? feature.high, feature.high, 0.06),
      centroid: lerp(averages.centroid ?? feature.centroid, feature.centroid, 0.07),
    };
    return { layout, horizon, pitch, farY, midY, groundY, subjectY, waterY, paintFeature };
  }

  redrawPainterTerrain() {
    const ctx = this.terrainCtx;
    const h = this.height;
    ctx.save();
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, this.terrainCanvas.width, this.terrainCanvas.height);
    ctx.restore();
    ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    const groups = [];
    let group = [];
    for (const point of this.painterTerrainPoints) {
      if (group.length && group[group.length - 1].sectionIndex !== point.sectionIndex) {
        const bridge = group[group.length - 1];
        groups.push(group);
        group = [bridge];
      }
      group.push(point);
    }
    if (group.length) groups.push(group);

    const traceSmooth = (points, yOf) => {
      const first = points[0];
      ctx.moveTo(first.x, yOf(first));
      if (points.length === 2) {
        ctx.lineTo(points[1].x, yOf(points[1]));
        return;
      }
      for (let i = 1; i < points.length - 1; i++) {
        const p = points[i];
        const next = points[i + 1];
        ctx.quadraticCurveTo(p.x, yOf(p), (p.x + next.x) * 0.5, (yOf(p) + yOf(next)) * 0.5);
      }
      const last = points[points.length - 1];
      ctx.lineTo(last.x, yOf(last));
    };

    const fillRibbon = (points, topOf, bottomOf, alpha, lightShift, saturationScale) => {
      if (points.length < 2) return;
      const sample = points[Math.floor(points.length * 0.55)];
      ctx.beginPath();
      traceSmooth(points, topOf);
      const reversed = Array.from(points).reverse();
      traceSmooth(reversed, bottomOf);
      ctx.closePath();
      ctx.fillStyle = this.paintColor(
        sample.paintFeature,
        (sample.layout.start + sample.layout.end) * 0.5,
        sample.sectionIndex,
        alpha,
        lightShift,
        saturationScale,
      );
      ctx.fill();
    };

    const strokeContour = (points, yOf, alpha, lightShift, width, saturationScale) => {
      if (points.length < 2) return;
      const sample = points[Math.floor(points.length * 0.55)];
      ctx.beginPath();
      traceSmooth(points, yOf);
      ctx.lineWidth = width;
      ctx.strokeStyle = this.paintColor(
        sample.paintFeature,
        (sample.layout.start + sample.layout.end) * 0.5,
        sample.sectionIndex,
        alpha,
        lightShift,
        saturationScale,
      );
      ctx.stroke();
    };

    for (const points of groups) {
      if (points.length < 2) continue;
      fillRibbon(points, (p) => p.farY, (p) => p.midY + h * 0.032, this.palette.dark ? 0.17 : 0.115, -0.13, 0.42);
      fillRibbon(points, (p) => p.midY, (p) => p.groundY + h * 0.020, this.palette.dark ? 0.27 : 0.18, -0.075, 0.52);
      fillRibbon(points, (p) => p.groundY, () => h * 1.03, this.palette.dark ? 0.46 : 0.34, -0.16, 0.48);
      strokeContour(points, (p) => p.farY, this.palette.dark ? 0.24 : 0.18, 0.08, 0.7, 0.38);
      strokeContour(points, (p) => p.midY, this.palette.dark ? 0.31 : 0.23, 0.04, 1.0, 0.40);
      strokeContour(points, (p) => p.groundY, this.palette.dark ? 0.23 : 0.17, 0.02, 1.3, 0.38);

      const sample = points[Math.floor(points.length * 0.5)];
      if (sample.layout.motif === 2) {
        const strands = 4 + Math.round(this.complexity * 2);
        for (let i = 0; i < strands; i++) {
          ctx.beginPath();
          traceSmooth(points, (p) => p.waterY + 5 + i * (5 + sample.paintFeature.low * 2));
          ctx.lineWidth = 0.45 + (1 - i / strands) * 0.85;
          ctx.strokeStyle = this.paintColor(sample.paintFeature, (sample.layout.start + sample.layout.end) * 0.5, sample.sectionIndex, this.palette.dark ? 0.17 : 0.12, 0.14, 0.30);
          ctx.stroke();
        }
      }
    }
  }

  drawMountainLandmark(x, metrics, feature, progress, stepIndex) {
    const ctx = this.ctx;
    const h = this.height;
    const layout = metrics.layout;
    const sectionWidth = Math.max(90, (layout.end - layout.start) * this.width);
    const width = clamp(sectionWidth * (0.46 + feature.energy * 0.12), 96, 250);
    const baseY = metrics.midY + h * 0.075;
    const peakLift = h * (0.12 + feature.energy * 0.13) * layout.ridgeScale;

    const drawMass = (layer, offsetX, offsetY, scale, alpha) => {
      const left = x - width * 0.52 * scale + offsetX;
      const right = x + width * 0.52 * scale + offsetX;
      const peak1X = x - width * (0.12 + randAt(this.seed, stepIndex + layer, 981) * 0.10) * scale + offsetX;
      const peak2X = x + width * (0.18 + randAt(this.seed, stepIndex + layer, 982) * 0.12) * scale + offsetX;
      const peak1Y = clamp(baseY - peakLift * (0.88 + randAt(this.seed, stepIndex + layer, 983) * 0.28) * scale + offsetY, h * 0.10, baseY - 38);
      const peak2Y = clamp(baseY - peakLift * (0.46 + randAt(this.seed, stepIndex + layer, 984) * 0.30) * scale + offsetY, h * 0.18, baseY - 22);
      const valleyY = lerp(peak1Y, baseY, 0.56 + randAt(this.seed, stepIndex + layer, 985) * 0.12);

      ctx.beginPath();
      ctx.moveTo(left, baseY + offsetY);
      ctx.bezierCurveTo(
        lerp(left, peak1X, 0.34), baseY - peakLift * 0.18 + offsetY,
        lerp(left, peak1X, 0.76), peak1Y + peakLift * 0.16,
        peak1X, peak1Y,
      );
      ctx.bezierCurveTo(
        lerp(peak1X, peak2X, 0.30), peak1Y + peakLift * 0.30,
        lerp(peak1X, peak2X, 0.62), valleyY,
        lerp(peak1X, peak2X, 0.68), valleyY,
      );
      ctx.bezierCurveTo(
        lerp(peak1X, peak2X, 0.80), valleyY - peakLift * 0.12,
        peak2X - width * 0.06, peak2Y + peakLift * 0.12,
        peak2X, peak2Y,
      );
      ctx.bezierCurveTo(
        lerp(peak2X, right, 0.38), peak2Y + peakLift * 0.25,
        lerp(peak2X, right, 0.74), baseY - peakLift * 0.10 + offsetY,
        right, baseY + offsetY,
      );
      ctx.closePath();
      const g = ctx.createLinearGradient(left, peak1Y, right, baseY);
      g.addColorStop(0, this.paintColor(metrics.paintFeature, progress, feature.sectionIndex, alpha * 0.72, 0.04 - layer * 0.04, 0.58));
      g.addColorStop(0.52, this.paintColor(metrics.paintFeature, progress, feature.sectionIndex, alpha, -0.02 - layer * 0.035, 0.62));
      g.addColorStop(1, this.paintColor(metrics.paintFeature, progress, feature.sectionIndex, alpha * 0.64, -0.12, 0.48));
      ctx.fillStyle = g;
      ctx.fill();
      return { left, right, peak1X, peak1Y, peak2X, peak2Y, baseY: baseY + offsetY };
    };

    drawMass(2, width * 0.10, 8, 0.82, this.palette.dark ? 0.24 : 0.17);
    const mass = drawMass(0, 0, 0, 1, this.palette.dark ? 0.47 : 0.34);
    drawMass(1, -width * 0.12, 12, 0.62, this.palette.dark ? 0.18 : 0.13);

    const contourCount = 4 + Math.round(this.complexity * 2);
    for (let i = 0; i < contourCount; i++) {
      const t = (i + 1) / (contourCount + 1);
      const y = lerp(Math.min(mass.peak1Y, mass.peak2Y), mass.baseY, t);
      const half = width * 0.44 * Math.sin(t * Math.PI * 0.82);
      ctx.beginPath();
      ctx.moveTo(x - half, y + (randAt(this.seed, stepIndex + i, 986) - 0.5) * 5);
      ctx.bezierCurveTo(
        x - half * 0.36, y - 4 - feature.mid * 6,
        x + half * 0.28, y + 4,
        x + half * 0.92, y + (randAt(this.seed, stepIndex + i, 987) - 0.5) * 5,
      );
      ctx.lineWidth = 0.55 + (1 - t) * 1.8;
      ctx.strokeStyle = this.paintColor(metrics.paintFeature, progress, feature.sectionIndex, this.palette.dark ? 0.26 : 0.20, 0.13 - t * 0.08, 0.40);
      ctx.stroke();
    }
  }

  drawLandscapeTree(x, groundY, scale, feature, progress, stepIndex, alpha = 0.78) {
    const ctx = this.ctx;
    const height = (24 + feature.mid * 42 + feature.onset * 22) * scale;
    const sway = (randAt(this.seed, stepIndex, 990) - 0.5) * height * 0.24;
    const topX = x + sway;
    const topY = groundY - height;
    const trunkWidth = 1.4 + feature.low * 3.2 + feature.energy * 1.8;
    this.drawBristleStroke({
      x1: x, y1: groundY, x2: topX, y2: topY,
      width: trunkWidth, curve: sway * 0.24,
      feature, progress, sectionIndex: feature.sectionIndex,
      alpha, stepIndex, channel: 991,
    });

    const branchCount = 3 + Math.floor(feature.high * 3 + this.complexity);
    for (let i = 0; i < branchCount; i++) {
      const t = 0.24 + i / Math.max(1, branchCount - 1) * 0.58;
      const bx = lerp(x, topX, t);
      const by = lerp(groundY, topY, t);
      const side = (i + stepIndex) % 2 ? -1 : 1;
      const length = height * (0.12 + randAt(this.seed, stepIndex * 11 + i, 992) * 0.18) * (0.75 + feature.high * 0.35);
      const ex = bx + side * length;
      const ey = by - length * (0.18 + randAt(this.seed, stepIndex * 13 + i, 993) * 0.48);
      ctx.beginPath();
      ctx.moveTo(bx, by);
      ctx.quadraticCurveTo(lerp(bx, ex, 0.52), lerp(by, ey, 0.52) - side * 3, ex, ey);
      ctx.lineWidth = Math.max(0.55, trunkWidth * (0.24 + (1 - t) * 0.16));
      ctx.strokeStyle = this.paintColor(feature, progress, feature.sectionIndex, alpha * 0.72, 0.10, 0.66);
      ctx.stroke();
    }

    if (feature.high > 0.28) {
      const crownCount = 3 + Math.floor(feature.high * 4);
      for (let i = 0; i < crownCount; i++) {
        const cx = topX + (randAt(this.seed, stepIndex * 17 + i, 994) - 0.5) * height * 0.38;
        const cy = topY + (randAt(this.seed, stepIndex * 19 + i, 995) - 0.5) * height * 0.18;
        const rx = 4 + feature.high * 10 + randAt(this.seed, stepIndex * 23 + i, 996) * 8;
        const ry = 2 + feature.high * 6 + randAt(this.seed, stepIndex * 29 + i, 997) * 5;
        this.organicBlobPath(cx, cy, rx, ry, stepIndex * 31 + i, 998, 8);
        ctx.fillStyle = this.paintColor(feature, progress, feature.sectionIndex, this.palette.dark ? 0.30 : 0.22, 0.12, 0.66);
        ctx.fill();
      }
    }
  }

  drawPavilionLandmark(x, metrics, feature, progress) {
    const ctx = this.ctx;
    const groundY = metrics.groundY;
    const scale = metrics.layout.subjectScale;
    const width = 54 * scale;
    const height = (48 + feature.mid * 32) * scale;
    const left = x - width * 0.5;
    const right = x + width * 0.5;
    const roofY = groundY - height;
    const color = this.paintColor(feature, progress, feature.sectionIndex, this.palette.dark ? 0.68 : 0.56, 0.06, 0.64);
    const fine = this.paintColor(feature, progress, feature.sectionIndex, this.palette.dark ? 0.46 : 0.36, 0.18, 0.50);

    ctx.lineCap = 'square';
    ctx.strokeStyle = color;
    ctx.lineWidth = 2.4 + feature.low * 2.0;
    ctx.beginPath();
    ctx.moveTo(left + width * 0.18, groundY);
    ctx.lineTo(left + width * 0.18, roofY + height * 0.20);
    ctx.moveTo(right - width * 0.18, groundY);
    ctx.lineTo(right - width * 0.18, roofY + height * 0.20);
    ctx.stroke();

    ctx.beginPath();
    ctx.moveTo(left - width * 0.14, roofY + height * 0.18);
    ctx.quadraticCurveTo(x, roofY - height * 0.12, right + width * 0.14, roofY + height * 0.18);
    ctx.quadraticCurveTo(x, roofY + height * 0.06, left - width * 0.14, roofY + height * 0.18);
    ctx.fillStyle = this.paintColor(feature, progress, feature.sectionIndex, this.palette.dark ? 0.42 : 0.32, -0.01, 0.70);
    ctx.fill();

    ctx.strokeStyle = fine;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(left + width * 0.08, groundY - height * 0.36);
    ctx.lineTo(right - width * 0.08, groundY - height * 0.36);
    ctx.moveTo(x, roofY + height * 0.16);
    ctx.lineTo(x, groundY);
    ctx.stroke();

    for (let i = 0; i < 3; i++) {
      ctx.beginPath();
      ctx.moveTo(left + width * (0.18 - i * 0.05), groundY + i * 4);
      ctx.lineTo(right - width * (0.18 - i * 0.05), groundY + i * 4);
      ctx.lineWidth = 1.2 + i * 0.8;
      ctx.strokeStyle = this.paintColor(feature, progress, feature.sectionIndex, 0.25, -0.14, 0.50);
      ctx.stroke();
    }
  }

  drawWaterLandmark(x, metrics, feature, progress, stepIndex) {
    const ctx = this.ctx;
    const radius = 12 + feature.high * 18 + feature.energy * 9;
    const y = clamp(metrics.farY - radius * 0.65, this.height * 0.12, metrics.waterY - radius * 1.7);
    const disc = ctx.createRadialGradient(x - radius * 0.25, y - radius * 0.25, radius * 0.08, x, y, radius);
    disc.addColorStop(0, this.paintColor(feature, progress, feature.sectionIndex, this.palette.dark ? 0.92 : 0.72, 0.28, 0.32));
    disc.addColorStop(0.72, this.paintColor(feature, progress, feature.sectionIndex, this.palette.dark ? 0.58 : 0.44, 0.18, 0.42));
    disc.addColorStop(1, this.paintColor(feature, progress, feature.sectionIndex, 0, 0.08, 0.42));
    ctx.fillStyle = disc;
    ctx.beginPath();
    ctx.arc(x, y, radius, 0, TAU);
    ctx.fill();

    const reflectionCount = 6 + Math.round(this.complexity * 3);
    for (let i = 0; i < reflectionCount; i++) {
      const ry = metrics.waterY + 4 + i * (4 + feature.low * 2);
      const half = radius * (0.22 + (1 - i / reflectionCount) * 0.76) * (0.7 + randAt(this.seed, stepIndex + i, 1001) * 0.5);
      ctx.beginPath();
      ctx.moveTo(x - half, ry);
      ctx.quadraticCurveTo(x, ry + (randAt(this.seed, stepIndex + i, 1002) - 0.5) * 3, x + half, ry);
      ctx.lineWidth = 0.7 + (1 - i / reflectionCount) * 2.2;
      ctx.strokeStyle = this.paintColor(feature, progress, feature.sectionIndex, this.palette.dark ? 0.42 : 0.30, 0.22, 0.34);
      ctx.stroke();
    }
  }

  drawGroveLandmark(x, metrics, feature, progress, stepIndex) {
    const count = 4 + Math.floor(feature.mid * 4 + this.complexity * 2);
    const spread = 42 + feature.energy * 42;
    for (let i = 0; i < count; i++) {
      const offset = (i / Math.max(1, count - 1) - 0.5) * spread + (randAt(this.seed, stepIndex + i, 1010) - 0.5) * 12;
      const scale = (0.58 + randAt(this.seed, stepIndex + i, 1011) * 0.64) * metrics.layout.subjectScale;
      this.drawLandscapeTree(x + offset, metrics.groundY + randAt(this.seed, stepIndex + i, 1012) * 5, scale, feature, progress, stepIndex * 17 + i, 0.72);
    }
  }

  drawSceneLandmark(x, metrics, feature, progress, stepIndex) {
    if (metrics.layout.motif === 0) this.drawMountainLandmark(x, metrics, feature, progress, stepIndex);
    else if (metrics.layout.motif === 1) this.drawGroveLandmark(x, metrics, feature, progress, stepIndex);
    else if (metrics.layout.motif === 2) this.drawWaterLandmark(x, metrics, feature, progress, stepIndex);
    else this.drawPavilionLandmark(x, metrics, feature, progress, stepIndex);
  }

  drawCloudStroke(x, y, length, feature, progress, stepIndex) {
    const ctx = this.ctx;
    const layers = 2 + Math.floor(feature.high * 2);
    for (let i = 0; i < layers; i++) {
      const yy = y + (i - (layers - 1) * 0.5) * (3 + feature.high * 4);
      const left = x - length * (0.55 + randAt(this.seed, stepIndex + i, 1020) * 0.18);
      const right = x + length * (0.45 + randAt(this.seed, stepIndex + i, 1021) * 0.18);
      ctx.beginPath();
      ctx.moveTo(left, yy);
      ctx.bezierCurveTo(lerp(left, right, 0.30), yy - 5 - feature.high * 7, lerp(left, right, 0.68), yy + 4, right, yy - 1);
      ctx.lineWidth = 1.2 + feature.high * 3.3 + i * 0.5;
      ctx.strokeStyle = this.paintColor(feature, progress, feature.sectionIndex, this.palette.dark ? 0.15 : 0.105, 0.20, 0.36);
      ctx.stroke();
    }
  }

  drawBirdMark(x, y, scale, feature, progress) {
    const ctx = this.ctx;
    ctx.beginPath();
    ctx.moveTo(x - scale, y);
    ctx.quadraticCurveTo(x - scale * 0.45, y - scale * 0.45, x, y);
    ctx.quadraticCurveTo(x + scale * 0.45, y - scale * 0.45, x + scale, y);
    ctx.lineWidth = Math.max(0.55, scale * 0.13);
    ctx.strokeStyle = this.paintColor(feature, progress, feature.sectionIndex, this.palette.dark ? 0.64 : 0.48, 0.12, 0.36);
    ctx.stroke();
  }

  generatePainterly(time, feature, stepIndex) {
    const ctx = this.ctx;
    const w = this.width;
    const h = this.height;
    if (w < 2 || h < 2) return;
    const marginX = Math.max(26, w * 0.04);
    const progress = clamp(time / this.duration);
    const x = marginX + progress * (w - marginX * 2);
    const metrics = this.painterlyMetrics(feature, progress);

    // Keep the landscape calm across analysis frames, including at section changes.
    if (this.painterPrev) {
      const sameSection = this.painterPrev.sectionIndex === feature.sectionIndex;
      const rate = sameSection ? 0.13 : 0.055;
      metrics.farY = lerp(this.painterPrev.farY, metrics.farY, rate);
      metrics.midY = lerp(this.painterPrev.midY, metrics.midY, rate * 1.20);
      metrics.groundY = lerp(this.painterPrev.groundY, metrics.groundY, rate * 0.90);
      metrics.waterY = lerp(this.painterPrev.waterY, metrics.waterY, rate);
    }
    const current = { x, progress, ...metrics, sectionIndex: feature.sectionIndex };
    this.painterTerrainPoints.push(current);
    this.redrawPainterTerrain();
    this.ctx = this.detailCtx;

    ctx.save();
    ctx.globalCompositeOperation = 'source-over';
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    // One compositional landmark per musical section gives the finished work a readable rhythm.
    if (!metrics.layout.landmarkDrawn && progress >= metrics.layout.focusProgress) {
      this.drawSceneLandmark(x, metrics, feature, progress, stepIndex);
      metrics.layout.landmarkDrawn = true;
      this.painterLastSubjectX = x;
      this.painterLastAccentX = x;
    }

    const sectionPixelWidth = Math.max(90, (metrics.layout.end - metrics.layout.start) * (w - marginX * 2));
    const cloudGap = clamp(sectionPixelWidth * 0.24, 72, 142) / (0.90 + this.complexity * 0.16);
    if (feature.high > 0.36 && x - this.painterLastCloudX > cloudGap) {
      const skyY = clamp(metrics.farY * (0.38 + randAt(this.seed, stepIndex, 1030) * 0.32), h * 0.10, h * 0.39);
      this.drawCloudStroke(x - 8, skyY, 20 + feature.high * 38, feature, progress, stepIndex);
      this.painterLastCloudX = x;
    }

    // Secondary subjects are deliberately scarce; silence and open sky are part of the painting.
    const subjectGap = clamp(sectionPixelWidth * 0.20, 62, 118) / (0.90 + this.complexity * 0.14);
    if (feature.mid > 0.42 && x - this.painterLastSubjectX > subjectGap) {
      const choice = randAt(this.seed, stepIndex, 1040);
      if (metrics.layout.motif === 1 || choice < 0.28) {
        const scale = (0.30 + feature.mid * 0.40) * metrics.layout.subjectScale;
        this.drawLandscapeTree(x, metrics.groundY + 2, scale, feature, progress, stepIndex, this.palette.dark ? 0.54 : 0.42);
      } else if (metrics.layout.motif === 2) {
        const count = 2 + Math.floor(feature.mid * 2);
        for (let i = 0; i < count; i++) {
          const yy = metrics.waterY + 5 + i * (5 + feature.low * 2);
          ctx.beginPath();
          ctx.moveTo(x - 13 - i * 2, yy);
          ctx.quadraticCurveTo(x, yy + (randAt(this.seed, stepIndex + i, 1041) - 0.5) * 3, x + 15 + i * 2, yy);
          ctx.lineWidth = 0.6 + feature.mid * 1.2;
          ctx.strokeStyle = this.paintColor(feature, progress, feature.sectionIndex, this.palette.dark ? 0.26 : 0.19, 0.14, 0.38);
          ctx.stroke();
        }
      } else {
        const rockW = 9 + feature.low * 14;
        const rockH = 5 + feature.mid * 11;
        this.organicBlobPath(x, metrics.groundY - rockH * 0.42, rockW, rockH, stepIndex, 1042, 9);
        ctx.fillStyle = this.paintColor(metrics.paintFeature, progress, feature.sectionIndex, this.palette.dark ? 0.30 : 0.22, -0.04, 0.50);
        ctx.fill();
      }
      this.painterLastSubjectX = x;
    }

    // Strong attacks become a small vocabulary of intentional marks. Varying the
    // gesture avoids the evenly spaced arches of a conventional visualiser.
    const accent = feature.onset * 0.92 + feature.energy * 0.08;
    const accentGap = clamp(sectionPixelWidth * 0.21, 68, 124) / (0.92 + this.complexity * 0.12);
    if (accent > 0.50 && x - this.painterLastAccentX > accentGap && randAt(this.seed, stepIndex, 1050) < accent * 0.72) {
      const choice = randAt(this.seed, stepIndex, 1051);
      const direction = randAt(this.seed, stepIndex, 1052) < 0.5 ? -1 : 1;
      const baseY = metrics.groundY - 3;
      if (choice < 0.30) {
        const length = 30 + feature.onset * 52 + feature.mid * 18;
        this.drawBristleStroke({
          x1: x - direction * length * 0.18,
          y1: baseY,
          x2: x + direction * length * 0.52,
          y2: clamp(metrics.subjectY - 8 - feature.onset * 14, h * 0.20, baseY - 20),
          width: 2.3 + feature.low * 4.6,
          curve: direction * (6 + feature.mid * 13),
          feature,
          progress,
          sectionIndex: feature.sectionIndex,
          alpha: this.palette.dark ? 0.62 : 0.48,
          stepIndex,
          channel: 1053,
        });
      } else if (choice < 0.58) {
        const length = 28 + feature.onset * 58;
        const y = lerp(metrics.midY, metrics.groundY, 0.54 + randAt(this.seed, stepIndex, 1054) * 0.28);
        this.drawKnifeMark(
          x + direction * length * 0.28,
          y,
          direction > 0 ? -0.06 - feature.mid * 0.18 : Math.PI + 0.06 + feature.mid * 0.18,
          length,
          3 + feature.low * 7,
          feature,
          progress,
          feature.sectionIndex,
          stepIndex,
          1055,
        );
      } else if (choice < 0.80) {
        const height = 24 + feature.onset * 42 + feature.mid * 16;
        this.drawBristleStroke({
          x1: x,
          y1: baseY,
          x2: x + direction * (4 + feature.high * 12),
          y2: baseY - height,
          width: 1.4 + feature.low * 3.4,
          curve: direction * (3 + feature.high * 8),
          feature,
          progress,
          sectionIndex: feature.sectionIndex,
          alpha: this.palette.dark ? 0.55 : 0.42,
          stepIndex,
          channel: 1056,
        });
      } else {
        const radius = 7 + feature.onset * 13;
        const y = lerp(metrics.midY, metrics.groundY, 0.45 + randAt(this.seed, stepIndex, 1057) * 0.32);
        this.organicBlobPath(x, y, radius * 1.25, radius * 0.65, stepIndex, 1058, 10);
        ctx.fillStyle = this.paintColor(feature, progress, feature.sectionIndex, this.palette.dark ? 0.38 : 0.28, 0.10, 0.56);
        ctx.fill();
        ctx.beginPath();
        ctx.arc(x, y, 1.2 + feature.onset * 2.4, 0, TAU);
        ctx.fillStyle = this.paintColor(feature, progress, feature.sectionIndex, this.palette.dark ? 0.72 : 0.52, 0.20, 0.40);
        ctx.fill();
      }
      this.painterLastAccentX = x;
    }

    const birdGap = 100 / (0.92 + this.complexity * 0.12);
    if (feature.high > 0.68 && x - this.painterLastBirdX > birdGap && randAt(this.seed, stepIndex, 1060) < feature.high * 0.58) {
      const birdCount = 1 + Math.floor(feature.high * 1.5);
      for (let i = 0; i < birdCount; i++) {
        const bx = x + (randAt(this.seed, stepIndex * 7 + i, 1061) - 0.5) * 22;
        const by = h * (0.14 + randAt(this.seed, stepIndex * 11 + i, 1062) * 0.20);
        this.drawBirdMark(bx, by, 2.4 + feature.high * 2.7 + i * 0.4, feature, progress);
      }
      this.painterLastBirdX = x;
    }

    // Low frequencies continue to leave a quiet horizontal ground stroke.
    if (feature.low > 0.54 && stepIndex % Math.max(8, Math.round(15 - this.complexity * 4)) === 0) {
      const length = 18 + feature.low * 34;
      this.drawBristleStroke({
        x1: x - length,
        y1: metrics.groundY + (randAt(this.seed, stepIndex, 1070) - 0.5) * 7,
        x2: x + length * 0.22,
        y2: metrics.groundY + (randAt(this.seed, stepIndex, 1071) - 0.5) * 5,
        width: 1.8 + feature.low * 4.8,
        curve: (randAt(this.seed, stepIndex, 1072) - 0.5) * 5,
        feature: metrics.paintFeature,
        progress,
        sectionIndex: feature.sectionIndex,
        alpha: this.palette.dark ? 0.34 : 0.25,
        stepIndex,
        channel: 1073,
      });
    }

    ctx.restore();
    this.composePainterly();
    this.painterPrev = current;
    this.lastSection = feature.sectionIndex;
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
    this.styleKey = 'sanctuary';
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
    this.floorCount = 0;
    this.foundationCount = 0;
    this.columnCount = 0;
    this.beamCount = 0;
    this.panelCount = 0;
    this.ornamentCount = 0;
    this.sanctuaryLayouts = [];
    this.sanctuaryHeading = 0;
    this.sanctuaryFloorAnchor = new THREE.Vector3(0, 0, 0);
    this.sanctuaryFloorWidth = 4.5;
    this.lastFloorStep = -999;
    this.lastBayStep = -999;
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

  configure({ analysis, duration, seed, palette, complexity, stepDuration, sculptureStyleKey = 'growth' }) {
    this.analysis = analysis;
    this.duration = Math.max(0.001, duration);
    this.seed = seed;
    this.palette = palette;
    this.complexity = complexity;
    this.styleKey = SCULPTURE_STYLES[sculptureStyleKey] ? sculptureStyleKey : 'growth';
    this.maxSteps = Math.ceil(duration / stepDuration) + 16;
    this.totalLength = this.styleKey === 'sanctuary'
      ? clamp(duration * 0.39, 48, 240)
      : clamp(duration * 0.47, 48, 260);
    this.buildSanctuaryLayouts();
    this.createInstances();
    this.applyPalette();
  }

  applyPalette() {
    const p = this.palette;
    this.scene.background = new THREE.Color(p.bg);
    this.scene.fog = null;
    this.grid.visible = true;
    this.grid.material.color.set(p.dark ? 0x3a4a62 : 0x60706c);
    this.grid.material.opacity = this.styleKey === 'sanctuary'
      ? (p.dark ? 0.055 : 0.075)
      : (p.dark ? 0.12 : 0.16);
    this.hemi.color.set(p.dark ? 0xc8e7ff : 0xffffff);
    this.hemi.groundColor.set(p.dark ? 0x24152d : 0x78827c);
    this.keyLight.intensity = this.styleKey === 'sanctuary' ? 2.05 : 1.72;
    this.rimLight.intensity = this.styleKey === 'sanctuary' ? 1.72 : 1.48;
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

    const boxGeometry = new THREE.BoxGeometry(1, 1, 1);
    const columnGeometry = new THREE.CylinderGeometry(0.78, 1, 1, 10, 1, false);
    const ornamentGeometry = new THREE.OctahedronGeometry(1, 0);
    const floorMaterial = new THREE.MeshStandardMaterial({ roughness: 0.54, metalness: 0.10 });
    const foundationMaterial = new THREE.MeshStandardMaterial({ roughness: 0.70, metalness: 0.04 });
    const columnMaterial = new THREE.MeshStandardMaterial({ roughness: 0.38, metalness: 0.18 });
    const beamMaterial = new THREE.MeshStandardMaterial({ roughness: 0.42, metalness: 0.17 });
    const panelMaterial = new THREE.MeshPhysicalMaterial({ roughness: 0.22, metalness: 0.12, transmission: 0.12, transparent: true, opacity: 0.62, side: THREE.DoubleSide });
    const ornamentMaterial = new THREE.MeshStandardMaterial({ roughness: 0.20, metalness: 0.34, emissiveIntensity: 0.20, flatShading: true });

    this.floors = new THREE.InstancedMesh(boxGeometry, floorMaterial, this.maxSteps);
    this.foundations = new THREE.InstancedMesh(boxGeometry, foundationMaterial, this.maxSteps);
    this.columns = new THREE.InstancedMesh(columnGeometry, columnMaterial, Math.ceil(this.maxSteps * 1.15));
    this.beams = new THREE.InstancedMesh(boxGeometry, beamMaterial, Math.ceil(this.maxSteps * 0.72));
    this.panels = new THREE.InstancedMesh(boxGeometry, panelMaterial, Math.ceil(this.maxSteps * 1.22));
    this.ornaments = new THREE.InstancedMesh(ornamentGeometry, ornamentMaterial, Math.ceil(this.maxSteps * 0.72));

    const growthMeshes = [this.trunk, this.branches, this.nodes, this.sparks];
    const sanctuaryMeshes = [this.floors, this.foundations, this.columns, this.beams, this.panels, this.ornaments];
    for (const mesh of [...growthMeshes, ...sanctuaryMeshes]) {
      mesh.count = 0;
      mesh.instanceMatrix.setUsage(THREE.DynamicDrawUsage);
      mesh.frustumCulled = false;
      this.root.add(mesh);
    }
    for (const mesh of growthMeshes) mesh.visible = this.styleKey === 'growth';
    for (const mesh of sanctuaryMeshes) mesh.visible = this.styleKey === 'sanctuary';
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
    this.floorCount = 0;
    this.foundationCount = 0;
    this.columnCount = 0;
    this.beamCount = 0;
    this.panelCount = 0;
    this.ornamentCount = 0;
    for (const mesh of [
      this.trunk, this.branches, this.nodes, this.sparks,
      this.floors, this.foundations, this.columns, this.beams, this.panels, this.ornaments,
    ]) {
      if (mesh) mesh.count = 0;
    }
    for (const mesh of this.sectionMeshes) {
      this.root.remove(mesh);
      this.disposeObject(mesh);
    }
    this.sectionMeshes.length = 0;
    this.lastSection = -1;
    this.lastBranchStep = -999;
    this.lastNodeStep = -999;
    this.lastBayStep = -999;
    this.lastFloorStep = -999;
    this.sanctuaryHeading = 0;
    this.sanctuaryFloorAnchor.set(0, 0, 0);
    this.sanctuaryFloorWidth = this.sanctuaryLayouts[0]?.width || 4.5;
    for (const layout of this.sanctuaryLayouts) layout.landmarkDrawn = false;
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

  buildSanctuaryLayouts() {
    this.sanctuaryLayouts = [];
    if (!this.analysis?.sections?.length) return;
    let heading = 0;
    let floorY = 0;
    const firstTurn = randAt(this.seed, 0, 1109) < 0.5 ? -1 : 1;
    for (let i = 0; i < this.analysis.sections.length; i++) {
      const section = this.analysis.sections[i];
      const a = section.averages;
      const direction = (i % 2 === 0 ? firstTurn : -firstTurn);
      const turn = direction * (0.14 + randAt(this.seed, i, 1110) * (0.24 + a.energy * 0.24));
      heading = clamp(heading + turn, -1.08, 1.08);
      floorY = clamp(floorY + (a.low - a.high) * 0.42 + (randAt(this.seed, i, 1111) - 0.5) * 0.62, -2.4, 2.4);
      let motif = Math.floor(randAt(this.seed, i, 1112) * 4);
      if (this.sanctuaryLayouts[i - 1]?.motif === motif) motif = (motif + 1 + Math.floor(randAt(this.seed, i, 1114) * 2)) % 4;
      this.sanctuaryLayouts.push({
        heading,
        floorY,
        width: 4.2 + a.low * 4.2 + a.energy * 1.15,
        height: 4.4 + a.mid * 4.9 + a.energy * 1.6,
        bayGap: Math.max(10, Math.round(20 - a.energy * 4.2 - this.complexity * 2.1)),
        motif,
        phase: randAt(this.seed, i, 1113) * TAU,
        centerProgress: clamp((section.start + section.end) * 0.5 / this.duration),
        colorFeature: {
          energy: a.energy,
          low: a.low,
          mid: a.mid,
          high: a.high,
          centroid: a.centroid,
          onset: 0.35,
          dominant: 0,
          sectionIndex: i,
          sectionPhase: 0.5,
        },
        landmarkDrawn: false,
      });
    }
  }

  sanctuaryFeature(layout, feature, liveMix = 0.10) {
    const base = layout.colorFeature || feature;
    return {
      energy: lerp(base.energy, feature.energy, liveMix),
      low: lerp(base.low, feature.low, liveMix),
      mid: lerp(base.mid, feature.mid, liveMix),
      high: lerp(base.high, feature.high, liveMix),
      centroid: lerp(base.centroid, feature.centroid, liveMix),
      onset: feature.onset,
      dominant: feature.dominant,
      sectionIndex: feature.sectionIndex,
      sectionPhase: feature.sectionPhase,
    };
  }

  setPathBoxInstance(mesh, index, a, b, width, height, color) {
    const { direction, midpoint, quaternion, scale, matrix } = this.tmp;
    direction.subVectors(b, a);
    const length = Math.max(0.0001, direction.length());
    direction.normalize();
    midpoint.addVectors(a, b).multiplyScalar(0.5);
    quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), direction);
    scale.set(Math.max(0.01, width), Math.max(0.01, height), length * 1.08);
    matrix.compose(midpoint, quaternion, scale);
    mesh.setMatrixAt(index, matrix);
    mesh.setColorAt(index, color);
  }

  createSanctuaryRibbonSegment(a, b, widthA, widthB, thickness, color, foundation = false) {
    const direction = b.clone().sub(a);
    direction.y = 0;
    if (direction.lengthSq() < 1e-8) return null;
    direction.normalize();
    const side = new THREE.Vector3(-direction.z, 0, direction.x);
    const topAL = a.clone().addScaledVector(side, widthA * 0.5);
    const topAR = a.clone().addScaledVector(side, -widthA * 0.5);
    const topBL = b.clone().addScaledVector(side, widthB * 0.5);
    const topBR = b.clone().addScaledVector(side, -widthB * 0.5);
    const down = new THREE.Vector3(0, thickness, 0);
    const bottomAL = topAL.clone().sub(down);
    const bottomAR = topAR.clone().sub(down);
    const bottomBL = topBL.clone().sub(down);
    const bottomBR = topBR.clone().sub(down);
    const points = [topAL, topAR, topBL, topBR, bottomAL, bottomAR, bottomBL, bottomBR];
    const positions = new Float32Array(points.length * 3);
    points.forEach((point, i) => point.toArray(positions, i * 3));
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
    geometry.setIndex([
      0, 2, 1, 1, 2, 3,
      4, 5, 6, 5, 7, 6,
      0, 4, 2, 2, 4, 6,
      1, 3, 5, 3, 7, 5,
      0, 1, 4, 1, 5, 4,
      2, 6, 3, 3, 6, 7,
    ]);
    geometry.computeVertexNormals();
    const material = new THREE.MeshStandardMaterial({
      color,
      roughness: foundation ? 0.72 : 0.46,
      metalness: foundation ? 0.035 : 0.12,
      emissive: color.clone().multiplyScalar(this.palette.dark ? (foundation ? 0.012 : 0.035) : 0.006),
      side: THREE.DoubleSide,
    });
    const mesh = new THREE.Mesh(geometry, material);
    mesh.castShadow = false;
    mesh.receiveShadow = true;
    this.root.add(mesh);
    this.sectionMeshes.push(mesh);
    return mesh;
  }

  setBeamInstance(mesh, index, a, b, thickness, depth, color) {
    const { direction, midpoint, quaternion, scale, matrix } = this.tmp;
    direction.subVectors(b, a);
    const length = Math.max(0.0001, direction.length());
    direction.normalize();
    midpoint.addVectors(a, b).multiplyScalar(0.5);
    quaternion.setFromUnitVectors(new THREE.Vector3(1, 0, 0), direction);
    scale.set(length, Math.max(0.01, thickness), Math.max(0.01, depth));
    matrix.compose(midpoint, quaternion, scale);
    mesh.setMatrixAt(index, matrix);
    mesh.setColorAt(index, color);
  }

  setBoxAtInstance(mesh, index, position, size, quaternion, color) {
    const { matrix } = this.tmp;
    matrix.compose(position, quaternion, size);
    mesh.setMatrixAt(index, matrix);
    mesh.setColorAt(index, color);
  }

  commitInstance(mesh, count) {
    mesh.count = count;
    mesh.instanceMatrix.needsUpdate = true;
    if (mesh.instanceColor) mesh.instanceColor.needsUpdate = true;
  }

  orientSanctuaryGroup(group, flatTangent) {
    const direction = flatTangent.clone();
    direction.y = 0;
    if (direction.lengthSq() < 1e-6) direction.set(1, 0, 0);
    direction.normalize();
    group.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), direction);
  }

  createSanctuaryBay(feature, layout, progress, stepIndex, width, height, flatTangent) {
    const group = new THREE.Group();
    group.position.copy(this.point);
    this.orientSanctuaryGroup(group, flatTangent);

    const stable = this.sanctuaryFeature(layout, feature, 0.10);
    const structuralColor = this.colorFor(stable, layout.centerProgress, -0.02).clone();
    const upperColor = this.colorFor(stable, layout.centerProgress, 0.12).clone();
    const glowColor = this.colorFor(feature, progress, 0.22).clone();
    const structureMaterial = new THREE.MeshStandardMaterial({
      color: structuralColor,
      roughness: 0.38,
      metalness: 0.20,
      emissive: structuralColor.clone().multiplyScalar(this.palette.dark ? 0.045 : 0.012),
    });
    const upperMaterial = new THREE.MeshStandardMaterial({
      color: upperColor,
      roughness: 0.30,
      metalness: 0.27,
      emissive: upperColor.clone().multiplyScalar(this.palette.dark ? 0.07 : 0.018),
    });
    const glassMaterial = new THREE.MeshPhysicalMaterial({
      color: glowColor.clone().offsetHSL(0, -0.10, 0.06),
      roughness: 0.16,
      metalness: 0.06,
      transmission: 0.16,
      transparent: true,
      opacity: this.palette.dark ? 0.42 : 0.32,
      side: THREE.DoubleSide,
    });
    const glowMaterial = new THREE.MeshStandardMaterial({
      color: glowColor,
      roughness: 0.18,
      metalness: 0.30,
      emissive: glowColor.clone().multiplyScalar(this.palette.dark ? 0.24 : 0.05),
      flatShading: true,
    });
    const boxGeometry = new THREE.BoxGeometry(1, 1, 1);

    const addBox = (position, scale, material = structureMaterial, rotation = null) => {
      const mesh = new THREE.Mesh(boxGeometry, material);
      mesh.position.copy(position);
      mesh.scale.copy(scale);
      if (rotation) mesh.rotation.set(rotation.x, rotation.y, rotation.z);
      group.add(mesh);
      return mesh;
    };
    const addBeam = (a, b, thicknessValue, depthValue, material = structureMaterial) => {
      const direction = b.clone().sub(a);
      const length = Math.max(0.001, direction.length());
      const mesh = new THREE.Mesh(boxGeometry, material);
      mesh.position.copy(a).add(b).multiplyScalar(0.5);
      mesh.quaternion.setFromUnitVectors(new THREE.Vector3(1, 0, 0), direction.normalize());
      mesh.scale.set(length, thicknessValue, depthValue);
      group.add(mesh);
      return mesh;
    };
    const addPillar = (x, pillarHeight, pillarWidth, z = 0, material = structureMaterial) => addBox(
      new THREE.Vector3(x, pillarHeight * 0.5, z),
      new THREE.Vector3(pillarWidth, pillarHeight, pillarWidth * 1.18),
      material,
    );

    const motif = (layout.motif + Math.floor(feature.sectionPhase * 3.2)) % 4;
    const pillarWidth = 0.18 + stable.low * 0.16;
    const half = width * 0.46;

    if (motif === 0) {
      // A rounded arch: the most architectural bay, used sparsely.
      addPillar(-half, height * 0.66, pillarWidth);
      addPillar(half, height * 0.66, pillarWidth);
      const curve = new THREE.CatmullRomCurve3([
        new THREE.Vector3(-half, height * 0.58, 0),
        new THREE.Vector3(-half * 0.78, height * 0.83, 0),
        new THREE.Vector3(-half * 0.38, height * 1.01, 0),
        new THREE.Vector3(0, height * 1.08, 0),
        new THREE.Vector3(half * 0.38, height * 1.01, 0),
        new THREE.Vector3(half * 0.78, height * 0.83, 0),
        new THREE.Vector3(half, height * 0.58, 0),
      ], false, 'catmullrom', 0.48);
      const arch = new THREE.Mesh(
        new THREE.TubeGeometry(curve, 34, 0.075 + stable.low * 0.055, 7, false),
        upperMaterial,
      );
      group.add(arch);
      const pendant = new THREE.Mesh(new THREE.OctahedronGeometry(0.22 + feature.high * 0.14, 0), glowMaterial);
      pendant.position.set(0, height * 0.70, 0);
      pendant.scale.y = 1.45 + feature.high * 0.55;
      group.add(pendant);
    } else if (motif === 1) {
      // A vaulted frame with a lifted roof line rather than another rectangular cage.
      const apex = new THREE.Vector3(0, height * 1.04, 0);
      const leftKnee = new THREE.Vector3(-half, height * 0.60, 0);
      const rightKnee = new THREE.Vector3(half, height * 0.60, 0);
      addPillar(-half, height * 0.60, pillarWidth * 0.90);
      addPillar(half, height * 0.60, pillarWidth * 0.90);
      addBeam(leftKnee, apex, 0.12 + feature.onset * 0.07, 0.16, upperMaterial);
      addBeam(apex, rightKnee, 0.12 + feature.onset * 0.07, 0.16, upperMaterial);
      addBeam(new THREE.Vector3(-half * 0.84, height * 0.62, 0), new THREE.Vector3(half * 0.84, height * 0.62, 0), 0.075, 0.12, structureMaterial);
      addBeam(new THREE.Vector3(-half * 1.16, 0, 0), new THREE.Vector3(-half, height * 0.58, 0), 0.08, 0.12, structureMaterial);
      addBeam(new THREE.Vector3(half * 1.16, 0, 0), new THREE.Vector3(half, height * 0.58, 0), 0.08, 0.12, structureMaterial);
    } else if (motif === 2) {
      // Thin translucent fins turn treble energy into light rather than more columns.
      const finCount = 4;
      for (let i = 0; i < finCount; i++) {
        const sideSign = i < 2 ? -1 : 1;
        const inner = i % 2;
        const x = sideSign * width * (inner ? 0.28 : 0.46);
        const finHeight = height * (0.54 + inner * 0.22 + stable.high * 0.12);
        addBox(
          new THREE.Vector3(x, finHeight * 0.50, (inner ? 0.10 : -0.08) * sideSign),
          new THREE.Vector3(0.055 + stable.high * 0.045, finHeight, 0.48 + stable.mid * 0.34),
          glassMaterial,
          new THREE.Euler(0, sideSign * (0.12 + inner * 0.10), sideSign * (0.07 + stable.high * 0.08)),
        );
      }
      addBeam(new THREE.Vector3(-half * 0.78, height * 0.84, 0), new THREE.Vector3(half * 0.78, height * 0.84, 0), 0.09, 0.13, upperMaterial);
      const crystal = new THREE.Mesh(new THREE.OctahedronGeometry(0.18 + stable.high * 0.16, 0), glowMaterial);
      crystal.position.set(0, height * 0.64, 0);
      crystal.scale.set(0.72, 1.55 + stable.high * 0.55, 0.72);
      group.add(crystal);
    } else {
      // A resonant halo opens the corridor into a room-like threshold.
      addPillar(-half * 0.92, height * 0.46, pillarWidth * 0.82);
      addPillar(half * 0.92, height * 0.46, pillarWidth * 0.82);
      const halo = new THREE.Mesh(new THREE.TorusGeometry(1, 0.055 + stable.high * 0.025, 8, 54), glassMaterial);
      halo.position.set(0, height * 0.62, 0);
      halo.scale.set(width * 0.30, height * 0.31, 1);
      group.add(halo);
      const core = new THREE.Mesh(new THREE.SphereGeometry(0.18 + feature.onset * 0.14, 18, 12), glowMaterial);
      core.position.set(0, height * 0.62, 0);
      group.add(core);
      addBeam(new THREE.Vector3(-half * 0.72, height * 0.24, 0), new THREE.Vector3(half * 0.72, height * 0.24, 0), 0.07, 0.11, structureMaterial);
    }

    this.root.add(group);
    this.sectionMeshes.push(group);
    const sideVector = new THREE.Vector3(-flatTangent.z, 0, flatTangent.x);
    this.expandBounds(this.point.clone().addScaledVector(sideVector, width * 0.66).add(new THREE.Vector3(0, height * 1.15, 0)));
    this.expandBounds(this.point.clone().addScaledVector(sideVector, -width * 0.66).add(new THREE.Vector3(0, -0.35, 0)));
  }

  createSanctuaryLandmark(feature, layout, progress, flatTangent, width, height) {
    const group = new THREE.Group();
    group.position.copy(this.point);
    this.orientSanctuaryGroup(group, flatTangent);
    const stable = this.sanctuaryFeature(layout, feature, 0.06);
    const baseColor = this.colorFor(stable, layout.centerProgress, -0.03).clone();
    const glowColor = this.colorFor(feature, progress, 0.24).clone();
    const baseMaterial = new THREE.MeshStandardMaterial({
      color: baseColor,
      roughness: 0.34,
      metalness: 0.24,
      emissive: baseColor.clone().multiplyScalar(this.palette.dark ? 0.06 : 0.015),
    });
    const glowMaterial = new THREE.MeshStandardMaterial({
      color: glowColor,
      roughness: 0.16,
      metalness: 0.28,
      emissive: glowColor.clone().multiplyScalar(this.palette.dark ? 0.34 : 0.07),
      flatShading: true,
    });
    const glassMaterial = new THREE.MeshPhysicalMaterial({
      color: glowColor.clone().offsetHSL(0, -0.12, 0.08),
      roughness: 0.12,
      metalness: 0.05,
      transmission: 0.22,
      transparent: true,
      opacity: this.palette.dark ? 0.48 : 0.35,
      side: THREE.DoubleSide,
    });
    const boxGeometry = new THREE.BoxGeometry(1, 1, 1);
    const plinth = new THREE.Mesh(boxGeometry, baseMaterial);
    plinth.position.set(0, 0.22, 0);
    plinth.scale.set(width * 0.48, 0.34 + stable.low * 0.22, 1.2 + stable.energy * 0.55);
    group.add(plinth);

    if (layout.motif === 0) {
      for (let i = 0; i < 3; i++) {
        const ring = new THREE.Mesh(new THREE.TorusGeometry(1, 0.045 + i * 0.012, 8, 58), i === 1 ? glowMaterial : glassMaterial);
        ring.position.set(0, height * (0.44 + i * 0.06), 0);
        ring.scale.set(width * (0.18 + i * 0.055), height * (0.20 + i * 0.045), 1);
        ring.rotation.y = (i - 1) * 0.32;
        ring.rotation.x = (i - 1) * 0.16;
        group.add(ring);
      }
      const core = new THREE.Mesh(new THREE.IcosahedronGeometry(0.30 + stable.energy * 0.24, 1), glowMaterial);
      core.position.set(0, height * 0.56, 0);
      group.add(core);
    } else if (layout.motif === 1) {
      const count = 5;
      for (let i = 0; i < count; i++) {
        const x = (i / (count - 1) - 0.5) * width * 0.58;
        const crystal = new THREE.Mesh(new THREE.OctahedronGeometry(0.24 + stable.high * 0.16, 0), i % 2 ? glassMaterial : glowMaterial);
        crystal.position.set(x, 0.72 + Math.sin(i * 1.7 + layout.phase) * 0.28 + (i % 2) * 0.45, 0);
        crystal.scale.set(0.72, 1.6 + stable.mid * 1.35 + (i % 3) * 0.34, 0.72);
        group.add(crystal);
      }
    } else if (layout.motif === 2) {
      const stem = new THREE.Mesh(boxGeometry, baseMaterial);
      stem.position.set(0, height * 0.34, 0);
      stem.scale.set(0.16 + stable.low * 0.12, height * 0.62, 0.18);
      group.add(stem);
      for (let i = 0; i < 5; i++) {
        const angle = -0.95 + i * 0.48;
        const length = width * (0.22 + (i % 2) * 0.07);
        const a = new THREE.Vector3(0, height * (0.34 + i * 0.075), 0);
        const b = new THREE.Vector3(Math.sin(angle) * length, a.y + Math.cos(angle) * length * 0.55, 0);
        const direction = b.clone().sub(a);
        const beam = new THREE.Mesh(boxGeometry, baseMaterial);
        beam.position.copy(a).add(b).multiplyScalar(0.5);
        beam.quaternion.setFromUnitVectors(new THREE.Vector3(1, 0, 0), direction.clone().normalize());
        beam.scale.set(direction.length(), 0.08, 0.10);
        group.add(beam);
        const pearl = new THREE.Mesh(new THREE.SphereGeometry(0.13 + stable.high * 0.08, 14, 10), glowMaterial);
        pearl.position.copy(b);
        group.add(pearl);
      }
    } else {
      for (let i = 0; i < 2; i++) {
        const halo = new THREE.Mesh(new THREE.TorusGeometry(1, 0.055 + i * 0.018, 8, 56), i ? glowMaterial : glassMaterial);
        halo.position.set(0, height * (0.50 + i * 0.05), 0);
        halo.scale.set(width * (0.26 + i * 0.08), height * (0.30 + i * 0.06), 1);
        group.add(halo);
      }
      const slab = new THREE.Mesh(boxGeometry, baseMaterial);
      slab.position.set(0, height * 0.34, 0.08);
      slab.scale.set(width * 0.18, height * 0.56, 0.16);
      group.add(slab);
    }

    const light = new THREE.PointLight(glowColor, this.palette.dark ? 1.5 : 0.75, 14, 2);
    light.position.set(0, height * 0.58, 1.2);
    group.add(light);
    this.root.add(group);
    this.sectionMeshes.push(group);
    const sideVector = new THREE.Vector3(-flatTangent.z, 0, flatTangent.x);
    this.expandBounds(this.point.clone().addScaledVector(sideVector, width * 0.52).add(new THREE.Vector3(0, height * 0.98, 0)));
    this.expandBounds(this.point.clone().addScaledVector(sideVector, -width * 0.52).add(new THREE.Vector3(0, -0.45, 0)));
  }

  createSanctuaryGate(feature, layout, progress) {
    const width = layout.width * (0.88 + feature.low * 0.22);
    const height = layout.height * (0.90 + feature.mid * 0.18);
    const group = new THREE.Group();
    group.position.copy(this.point);
    const flatTangent = this.tangent.clone();
    flatTangent.y = 0;
    if (flatTangent.lengthSq() < 1e-6) flatTangent.set(1, 0, 0);
    flatTangent.normalize();
    group.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), flatTangent);

    const color = this.colorFor(feature, progress, 0.08).clone();
    const material = new THREE.MeshStandardMaterial({ color, roughness: 0.36, metalness: 0.20, emissive: color.clone().multiplyScalar(this.palette.dark ? 0.10 : 0.025) });
    const glassMaterial = new THREE.MeshPhysicalMaterial({ color: color.clone().offsetHSL(0, -0.12, 0.12), roughness: 0.18, metalness: 0.08, transmission: 0.10, transparent: true, opacity: 0.54, side: THREE.DoubleSide });
    const pillarGeometry = new THREE.BoxGeometry(1, 1, 1);
    const pillarW = 0.22 + feature.low * 0.18;
    for (const side of [-1, 1]) {
      const pillar = new THREE.Mesh(pillarGeometry, material);
      pillar.position.set(side * width * 0.48, height * 0.5, 0);
      pillar.scale.set(pillarW, height, pillarW * 1.45);
      group.add(pillar);
    }
    const lintel = new THREE.Mesh(pillarGeometry, material);
    lintel.position.set(0, height, 0);
    lintel.scale.set(width + pillarW * 2, 0.20 + feature.onset * 0.18, 0.28 + feature.low * 0.16);
    group.add(lintel);

    const halo = new THREE.Mesh(new THREE.TorusGeometry(1, 0.045 + feature.high * 0.028, 8, 46), glassMaterial);
    halo.position.set(0, height * 0.56, 0.05);
    halo.scale.set(width * 0.38, height * 0.36, 1);
    group.add(halo);

    this.root.add(group);
    this.sectionMeshes.push(group);
    const sideVector = new THREE.Vector3(-flatTangent.z, 0, flatTangent.x);
    this.expandBounds(this.point.clone().addScaledVector(sideVector, width * 0.62).add(new THREE.Vector3(0, height * 1.12, 0)));
    this.expandBounds(this.point.clone().addScaledVector(sideVector, -width * 0.62).add(new THREE.Vector3(0, -0.6, 0)));
  }

  generate(time, feature, stepIndex) {
    if (this.styleKey === 'sanctuary') this.generateSanctuary(time, feature, stepIndex);
    else this.generateGrowth(time, feature, stepIndex);
  }

  generateSanctuary(time, feature, stepIndex) {
    if (!this.floors || this.floorCount >= this.floors.instanceMatrix.count) return;
    const progress = clamp(time / this.duration);
    const dx = this.totalLength / Math.max(1, this.maxSteps - 1);
    const layout = this.sanctuaryLayouts[feature.sectionIndex] || {
      heading: 0, floorY: 0, width: 5, height: 5.5, bayGap: 7, motif: 0, phase: 0,
    };

    const headingNoise = (valueNoise1D(progress * (3.4 + this.complexity), this.seed, 1120 + feature.sectionIndex) - 0.5) * (0.12 + feature.high * 0.08);
    const targetHeading = layout.heading + headingNoise + (feature.centroid - 0.5) * 0.045;
    let deltaHeading = targetHeading - this.sanctuaryHeading;
    while (deltaHeading > Math.PI) deltaHeading -= TAU;
    while (deltaHeading < -Math.PI) deltaHeading += TAU;
    this.sanctuaryHeading += deltaHeading * (0.048 + feature.onset * 0.026);

    const targetY = layout.floorY + (feature.low - feature.high) * 0.22 + (valueNoise1D(progress * 5.2, this.seed, 1121) - 0.5) * 0.24;
    const stepY = (targetY - this.point.y) * 0.055;
    this.prevPoint.copy(this.point);
    this.point.add(new THREE.Vector3(Math.cos(this.sanctuaryHeading) * dx, stepY, Math.sin(this.sanctuaryHeading) * dx));
    this.tangent.subVectors(this.point, this.prevPoint).normalize();

    const flatTangent = this.tangent.clone();
    flatTangent.y = 0;
    if (flatTangent.lengthSq() < 1e-6) flatTangent.set(1, 0, 0);
    flatTangent.normalize();
    const side = new THREE.Vector3(-flatTangent.z, 0, flatTangent.x);
    const width = layout.width * (0.78 + feature.low * 0.30 + feature.energy * 0.08);
    const thickness = 0.14 + feature.low * 0.22 + feature.energy * 0.08;
    const stableFeature = this.sanctuaryFeature(layout, feature, 0.08);
    const floorColor = this.colorFor(stableFeature, layout.centerProgress, -0.12).clone();
    const foundationColor = this.colorFor(stableFeature, layout.centerProgress, -0.22).clone();

    const floorStride = Math.max(4, Math.round(8 - this.complexity * 2));
    const shouldLayFloor = stepIndex === 0 || stepIndex - this.lastFloorStep >= floorStride || progress > 0.997;
    if (shouldLayFloor && this.floorCount < this.maxSteps) {
      const floorA = this.sanctuaryFloorAnchor.clone();
      const floorB = this.point.clone();
      this.createSanctuaryRibbonSegment(
        floorA, floorB,
        this.sanctuaryFloorWidth, width,
        thickness, floorColor, false,
      );
      this.floorCount++;

      const depth = 0.34 + feature.low * 0.72;
      const foundationA = floorA.clone().add(new THREE.Vector3(0, -thickness, 0));
      const foundationB = floorB.clone().add(new THREE.Vector3(0, -thickness, 0));
      const baseScale = 0.80 + feature.low * 0.10;
      this.createSanctuaryRibbonSegment(
        foundationA, foundationB,
        this.sanctuaryFloorWidth * baseScale, width * baseScale,
        depth, foundationColor, true,
      );
      this.foundationCount++;

      this.sanctuaryFloorAnchor.copy(this.point);
      this.sanctuaryFloorWidth = width;
      this.lastFloorStep = stepIndex;
    }

    const leftEdge = this.point.clone().addScaledVector(side, width * 0.54);
    const rightEdge = this.point.clone().addScaledVector(side, -width * 0.54);
    this.expandBounds(leftEdge.clone().add(new THREE.Vector3(0, -1.4, 0)));
    this.expandBounds(rightEdge.clone().add(new THREE.Vector3(0, -1.4, 0)));

    if (feature.sectionIndex !== this.lastSection) {
      this.createSanctuaryGate(feature, layout, progress);
      this.lastSection = feature.sectionIndex;
      this.lastBayStep = stepIndex - layout.bayGap;
    }

    if (!layout.landmarkDrawn && feature.sectionPhase >= 0.48) {
      const landmarkHeight = layout.height * (0.92 + feature.mid * 0.14);
      this.createSanctuaryLandmark(feature, layout, progress, flatTangent, width, landmarkHeight);
      layout.landmarkDrawn = true;
      this.lastBayStep = stepIndex;
    }

    const bayGap = Math.max(9, layout.bayGap - (feature.onset > 0.78 ? 1 : 0));
    const baySignal = feature.onset * 0.56 + feature.mid * 0.24 + feature.energy * 0.20;
    if (stepIndex - this.lastBayStep >= bayGap && (baySignal > 0.34 || stepIndex % bayGap === 0)) {
      const height = layout.height * (0.78 + feature.mid * 0.26 + feature.energy * 0.08);
      this.createSanctuaryBay(feature, layout, progress, stepIndex, width, height, flatTangent);
      this.lastBayStep = stepIndex;
    }

    const panelSignal = feature.high * 0.78 + feature.onset * 0.22;
    if (panelSignal > 0.66 && randAt(this.seed, stepIndex, 1130) < panelSignal * (0.07 + this.complexity * 0.035)) {
      const count = feature.high > 0.78 && this.complexity > 1.05 ? 2 : 1;
      for (let i = 0; i < count && this.panelCount < this.panels.instanceMatrix.count; i++) {
        const sideSign = (i + stepIndex) % 2 ? -1 : 1;
        const panelPos = this.point.clone()
          .addScaledVector(side, sideSign * width * (0.31 + randAt(this.seed, stepIndex + i, 1131) * 0.17))
          .add(new THREE.Vector3(0, 0.9 + feature.high * 2.6 + randAt(this.seed, stepIndex + i, 1132) * 1.2, 0));
        const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), flatTangent);
        q.multiply(new THREE.Quaternion().setFromAxisAngle(new THREE.Vector3(0, 1, 0), sideSign * (0.12 + randAt(this.seed, stepIndex + i, 1133) * 0.28)));
        const size = new THREE.Vector3(0.06 + feature.high * 0.10, 0.62 + feature.high * 1.55, 0.52 + feature.mid * 0.72);
        this.setBoxAtInstance(this.panels, this.panelCount, panelPos, size, q, this.colorFor(stableFeature, layout.centerProgress, 0.18).clone());
        this.panelCount++;
        this.expandBounds(panelPos.clone().add(new THREE.Vector3(0, size.y, 0)));
      }
      this.commitInstance(this.panels, this.panelCount);
    }

    if (feature.low > 0.58 && feature.onset > 0.58 && randAt(this.seed, stepIndex, 1140) < feature.onset * 0.34 && this.panelCount < this.panels.instanceMatrix.count) {
      const altarPos = this.point.clone().add(new THREE.Vector3(0, 0.28 + feature.low * 0.22, 0));
      const q = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 0, 1), flatTangent);
      const size = new THREE.Vector3(width * 0.34, 0.34 + feature.low * 0.42, 0.52 + feature.onset * 0.72);
      this.setBoxAtInstance(this.panels, this.panelCount, altarPos, size, q, this.colorFor(stableFeature, layout.centerProgress, 0.04).clone());
      this.panelCount++;
      this.commitInstance(this.panels, this.panelCount);
      this.expandBounds(altarPos.clone().add(new THREE.Vector3(0, size.y, 0)));
    }

    this.tip.position.copy(this.point).add(new THREE.Vector3(0, 0.35, 0));
    this.tip.scale.setScalar(0.78 + feature.onset * 1.30);
    this.tip.material.color.copy(this.colorFor(feature, progress, 0.24));
    this.tipLight.position.copy(this.point).add(new THREE.Vector3(0, 3.0, 2.2));
    this.tipLight.color.copy(this.tip.material.color);
  }

  generateGrowth(time, feature, stepIndex) {
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
    const fitDistance = Math.max(13, radius / Math.sin(fov / 2) * (this.styleKey === 'sanctuary' ? 0.79 : 0.96));
    let desiredPosition;
    let desiredTarget;
    if (this.cameraMode === 'overview') {
      const viewDirection = new THREE.Vector3(0.82, 0.46, 1.0).normalize();
      desiredPosition = center.clone().addScaledVector(viewDirection, fitDistance * 0.92);
      desiredTarget = center;
    } else {
      // Ahead of the growth tangent, looking back through the current tip and the accumulated work.
      const ahead = this.tangent.clone().multiplyScalar(fitDistance * (this.styleKey === 'sanctuary' ? 0.46 : 0.64));
      const up = new THREE.Vector3(0, fitDistance * (this.styleKey === 'sanctuary' ? 0.18 : 0.20) + 2.0, 0);
      const lateral = new THREE.Vector3(-this.tangent.z, 0, this.tangent.x).multiplyScalar(fitDistance * (this.styleKey === 'sanctuary' ? 0.28 : 0.34));
      desiredPosition = this.point.clone().add(ahead).add(up).add(lateral);
      desiredTarget = center.clone().lerp(this.point, this.styleKey === 'sanctuary' ? 0.64 : 0.48);
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
  sculptureStyleSelect: document.querySelector('#sculptureStyleSelect'),
  sculptureStyleSetting: document.querySelector('#sculptureStyleSetting'),
  sculptureStyleDescription: document.querySelector('#sculptureStyleDescription'),
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
  sculptureStyleKey: 'sanctuary',
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
    sculptureStyleKey: state.sculptureStyleKey,
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

function updateSculptureStyleUI() {
  const style = SCULPTURE_STYLES[state.sculptureStyleKey] || SCULPTURE_STYLES.growth;
  if (els.sculptureStyleSelect) els.sculptureStyleSelect.value = state.sculptureStyleKey;
  if (els.sculptureStyleDescription) els.sculptureStyleDescription.textContent = style.description;
  els.stage.dataset.sculptureStyle = state.sculptureStyleKey;
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
  const sculptureStyle = SCULPTURE_STYLES[state.sculptureStyleKey] || SCULPTURE_STYLES.growth;
  if (state.mode === '2d') {
    els.modeHint.textContent = paintingStyle.hint;
    els.stylePill.textContent = paintingStyle.label;
    els.mappingRules.innerHTML = paintingStyle.mappingHtml;
    els.paintingStyleSetting.hidden = false;
    els.sculptureStyleSetting.hidden = true;
  } else {
    els.modeHint.textContent = sculptureStyle.hint;
    els.stylePill.textContent = sculptureStyle.label;
    els.mappingRules.innerHTML = sculptureStyle.mappingHtml;
    els.paintingStyleSetting.hidden = true;
    els.sculptureStyleSetting.hidden = false;
  }
  els.cameraBtn.hidden = state.mode !== '3d' || !webglAvailable;
  els.exportBtn.querySelector('span').textContent = state.mode === '2d' ? '导出画作' : '导出视图';
  updatePaintingStyleUI();
  updateSculptureStyleUI();
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
    const style = SCULPTURE_STYLES[state.sculptureStyleKey] || SCULPTURE_STYLES.growth;
    downloadBlob(blob, `${baseName}_3D_${style.exportName}_${els.seedValue.textContent}.png`);
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

  els.sculptureStyleSelect.addEventListener('change', async () => {
    const nextStyle = els.sculptureStyleSelect.value;
    if (!SCULPTURE_STYLES[nextStyle] || nextStyle === state.sculptureStyleKey) return;
    const wasPlaying = state.isPlaying;
    if (wasPlaying) pause();
    state.sculptureStyleKey = nextStyle;
    updateModeUI();
    if (state.analysis) {
      configureVisuals();
      await rebuildTo(state.currentTime, state.duration > 600);
    } else {
      sculpture.styleKey = nextStyle;
      sculpture.applyPalette();
    }
    if (wasPlaying) await play();
    toast(`已切换到${SCULPTURE_STYLES[nextStyle].label}`);
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
  updateSculptureStyleUI();
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
    const requested2d = params.get('style2d');
    const requested3d = params.get('style3d');
    if (requested2d && PAINTING_STYLES[requested2d]) state.paintingStyleKey = requested2d;
    if (requested3d && SCULPTURE_STYLES[requested3d]) state.sculptureStyleKey = requested3d;
    state.mode = mode;
    updateModeUI();
    await loadDemo(false, mode === '2d' ? 0.94 : 0.88);
  }
}

init().catch(showError);

export function installStyle3(runtime) {
  const {
    THREE,
    TAU,
    clamp,
    lerp,
    smoothstep,
    randAt,
    valueNoise1D,
    PAINTING_STYLES,
    SCULPTURE_STYLES,
    CanvasPainting,
    Sculpture3D,
    state,
    painting,
    sculpture,
    els,
  } = runtime;

  PAINTING_STYLES.journey = {
    order: 3,
    name: '行旅画卷',
    label: '风格 3 · 行旅画卷',
    description: '音乐沿一条隐性的行旅脉络前进，不再把画面直接平铺出来；山体、水气、树林、亭廊、花簇与灯火会在路径经过时逐步被引入画卷。',
    hint: '风格 3 · 音乐像一次行旅，沿途不断引出景物与留白',
    mappingHtml: '时间 → 沿行旅路径前进并展开画卷<br>低频 → 地势、水岸、山体与前景重量<br>中频 / 音高 → 路径起伏、景物落点与主体走势<br>高频 → 云气、飞白、灯火与细部纹理<br>瞬态 → 驿站、树影、花簇、鸟迹与重笔<br>段落 → 行进方向、章节景象与留白节奏',
    exportName: '风格3_行旅画卷',
  };

  SCULPTURE_STYLES.citadel = {
    order: 3,
    name: '共鸣遗城',
    label: '风格 3 · 共鸣遗城',
    description: '在殿堂式空间章节上继续扩展台阶、塔楼、回廊、能量池、遗碑、浮石、悬灯与仪式环，让音乐逐步唤醒一座拥有多种建造语法的遗城。',
    hint: '风格 3 · 音乐逐步唤醒一座遗城，廊道、塔楼与仪式场并存',
    mappingHtml: '时间 → 沿城迹推进并逐步建造章节<br>低频 → 平台、台阶、桥面与地基层级<br>中频 → 塔楼、柱列、门洞、回廊与空间高度<br>高频 → 悬灯、浮石、晶体、光环与顶饰<br>瞬态 → 遗碑、祭坛、门楼、钟片与空间重音<br>段落 → 城区布局、章节类型与行进转向',
    exportName: '风格3_共鸣遗城',
  };

  const ensureOption = (select, value, label) => {
    if (!select || Array.from(select.options).some((option) => option.value === value)) return;
    const option = document.createElement('option');
    option.value = value;
    option.textContent = label;
    select.appendChild(option);
  };
  ensureOption(els.paintingStyleSelect, 'journey', PAINTING_STYLES.journey.label);
  ensureOption(els.sculptureStyleSelect, 'citadel', SCULPTURE_STYLES.citadel.label);

  installJourneyPainting();
  installCitadelSculpture();

  state.paintingStyleKey = 'journey';
  state.sculptureStyleKey = 'citadel';
  painting.styleKey = 'journey';
  sculpture.styleKey = 'citadel';

  function installJourneyPainting() {
    const proto = CanvasPainting.prototype;
    if (proto.__journeyInstalled) return;
    proto.__journeyInstalled = true;

    const originalConfigure = proto.configure;
    const originalResize = proto.resize;
    const originalReset = proto.reset;
    const originalGenerate = proto.generate;

    proto.configure = function configureWithJourney(config) {
      originalConfigure.call(this, config);
      this.buildJourneyPlan();
    };

    proto.resize = function resizeWithJourney() {
      const changed = originalResize.call(this);
      if (this.styleKey === 'journey') this.ctx = this.detailCtx;
      return changed;
    };

    proto.reset = function resetWithJourney() {
      if (this.styleKey === 'journey') this.resetJourney();
      else originalReset.call(this);
    };

    proto.generate = function generateWithJourney(time, feature, stepIndex) {
      if (this.styleKey === 'journey') this.generateJourney(time, feature, stepIndex);
      else originalGenerate.call(this, time, feature, stepIndex);
    };

    proto.buildJourneyPlan = function buildJourneyPlan() {
      this.journeyPlan = null;
      if (!this.analysis?.sections?.length) return;
      const anchors = [];
      let pathY = clamp(0.55 + (randAt(this.seed, 0, 1300) - 0.5) * 0.20, 0.32, 0.74);
      let direction = randAt(this.seed, 0, 1301) < 0.5 ? -1 : 1;
      anchors.push({ progress: 0, y: pathY, motif: 0, side: direction, chapter: 0, openness: 0.62 });

      this.analysis.sections.forEach((section, index) => {
        const a = section.averages;
        const start = clamp(section.start / this.duration);
        const end = clamp(section.end / this.duration);
        const center = (start + end) * 0.5;
        direction *= randAt(this.seed, index, 1302) > 0.27 ? -1 : 1;
        const pitchPull = (0.5 - a.centroid) * 0.13;
        const bassPull = (a.low - a.high) * 0.08;
        const drift = (randAt(this.seed, index, 1303) - 0.5) * (0.13 + a.energy * 0.06);
        pathY = clamp(pathY + direction * (0.045 + a.mid * 0.055) + pitchPull + bassPull + drift, 0.18, 0.82);

        let motif;
        const roll = randAt(this.seed, index, 1304);
        if (a.low > a.high + 0.15) motif = 0;
        else if (a.high > a.low + 0.18) motif = 2;
        else if (a.mid > 0.60) motif = 3;
        else motif = roll < 0.28 ? 0 : roll < 0.55 ? 1 : roll < 0.78 ? 2 : 3;

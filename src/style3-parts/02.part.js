        if (anchors[anchors.length - 1]?.motif === motif) motif = (motif + 1 + Math.floor(randAt(this.seed, index, 1305) * 2)) % 4;

        anchors.push({
          progress: center,
          start,
          end,
          y: pathY,
          motif,
          side: randAt(this.seed, index, 1306) < 0.5 ? -1 : 1,
          chapter: index,
          openness: clamp(0.30 + a.high * 0.46 - a.low * 0.12 + randAt(this.seed, index, 1307) * 0.18, 0.22, 0.88),
          scale: 0.76 + a.energy * 0.42 + a.mid * 0.18,
          phase: randAt(this.seed, index, 1308) * TAU,
        });
      });

      anchors.push({
        progress: 1,
        y: clamp(pathY + (randAt(this.seed, 0, 1309) - 0.5) * 0.08, 0.18, 0.82),
        motif: anchors[anchors.length - 1]?.motif || 0,
        side: -1,
        chapter: this.analysis.sections.length - 1,
        openness: 0.66,
      });
      anchors.sort((a, b) => a.progress - b.progress);
      this.journeyPlan = { anchors };
    };

    proto.resetJourney = function resetJourney() {
      const { width: w, height: h, palette } = this;
      for (const context of [this.terrainCtx, this.detailCtx]) {
        context.save();
        context.setTransform(1, 0, 0, 1, 0, 0);
        context.clearRect(0, 0, this.canvas.width, this.canvas.height);
        context.restore();
        context.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      }

      const ctx = this.baseCtx;
      ctx.save();
      ctx.setTransform(this.dpr, 0, 0, this.dpr, 0, 0);
      const base = ctx.createLinearGradient(0, 0, w, h);
      base.addColorStop(0, palette.bg);
      base.addColorStop(0.48, palette.bg2);
      base.addColorStop(1, palette.bg);
      ctx.fillStyle = base;
      ctx.fillRect(0, 0, w, h);

      const distant = ctx.createLinearGradient(0, h * 0.16, 0, h * 0.88);
      distant.addColorStop(0, 'rgba(0,0,0,0)');
      distant.addColorStop(0.55, this.paintColor({ energy: 0.28, low: 0.34, mid: 0.34, high: 0.28, centroid: 0.36 }, 0.5, 0, palette.dark ? 0.036 : 0.026, 0.08, 0.42));
      distant.addColorStop(1, this.paintColor({ energy: 0.32, low: 0.46, mid: 0.30, high: 0.18, centroid: 0.28 }, 0.5, 0, palette.dark ? 0.13 : 0.08, -0.14, 0.44));
      ctx.fillStyle = distant;
      ctx.fillRect(0, 0, w, h);

      const fibreCount = Math.min(620, Math.max(220, Math.round((w * h) / 2700)));
      ctx.lineCap = 'round';
      for (let i = 0; i < fibreCount; i++) {
        const x = randAt(this.seed, i, 1310) * w;
        const y = randAt(this.seed, i, 1311) * h;
        const length = 2 + randAt(this.seed, i, 1312) * 20;
        const angle = (randAt(this.seed, i, 1313) - 0.5) * 0.24;
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + Math.cos(angle) * length, y + Math.sin(angle) * length);
        ctx.lineWidth = 0.25 + randAt(this.seed, i, 1314) * 0.46;
        ctx.strokeStyle = palette.dark
          ? `rgba(255,255,255,${0.009 + randAt(this.seed, i, 1315) * 0.014})`
          : `rgba(25,34,31,${0.013 + randAt(this.seed, i, 1315) * 0.018})`;
        ctx.stroke();
      }
      ctx.restore();

      this.ctx = this.detailCtx;
      this.journeyPrev = null;
      this.journeyLastSubjectX = -Infinity;
      this.journeyLastCloudX = -Infinity;
      this.journeyLastBirdX = -Infinity;
      this.journeyLastAccentX = -Infinity;
      this.lastSection = -1;
      this.composePainterly();
    };

    proto.journeyPoint = function journeyPoint(progress, feature) {
      const marginX = Math.max(26, this.width * 0.04);
      const x = marginX + progress * (this.width - marginX * 2);
      const anchors = this.journeyPlan?.anchors;
      if (!anchors?.length) return { x, y: this.height * 0.52, angle: 0, anchor: null };
      let a = anchors[0];
      let b = anchors[anchors.length - 1];
      for (let i = 1; i < anchors.length; i++) {
        if (progress <= anchors[i].progress) {
          a = anchors[i - 1];
          b = anchors[i];
          break;
        }
      }
      const local = clamp((progress - a.progress) / Math.max(0.0001, b.progress - a.progress));
      const eased = smoothstep(0, 1, local);
      const pitch = feature.dominant > 1
        ? clamp((Math.log2(Math.max(55, feature.dominant)) - Math.log2(55)) / 6)
        : feature.centroid;
      const musicalLift = (0.5 - pitch) * 0.085 + (feature.mid - feature.low) * 0.045;
      const breathing = Math.sin(progress * TAU * 1.25 + (a.phase || 0)) * (0.018 + feature.energy * 0.012);
      const noise = (valueNoise1D(progress * (5.0 + this.complexity * 2.2), this.seed, 1320) - 0.5) * 0.060;
      const yNorm = clamp(lerp(a.y, b.y, eased) + musicalLift + breathing + noise, 0.10, 0.91);

      const epsilon = 0.008;
      const next = clamp(progress + epsilon, 0, 1);
      const nextX = marginX + next * (this.width - marginX * 2);
      let na = a;
      let nb = b;
      for (let i = 1; i < anchors.length; i++) {
        if (next <= anchors[i].progress) {
          na = anchors[i - 1];
          nb = anchors[i];

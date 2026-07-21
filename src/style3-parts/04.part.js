        this.drawWaterLandmark(x, metrics, feature, progress, stepIndex + 1600);
      } else {
        this.drawPavilionLandmark(x, metrics, feature, progress, stepIndex + 1700);
      }

      if (force || feature.onset > 0.58) {
        const treeSide = direction * -1;
        this.drawLandscapeTree(
          point.x + normalX * treeSide * (18 + feature.low * 18),
          clamp(point.y + this.height * 0.12, this.height * 0.34, this.height * 0.92),
          0.58 + feature.energy * 0.42,
          feature,
          progress,
          stepIndex + 1800,
          this.palette.dark ? 0.72 : 0.62,
        );
      }
    };

    proto.generateJourney = function generateJourney(time, feature, stepIndex) {
      const w = this.width;
      const h = this.height;
      if (w < 2 || h < 2 || !this.journeyPlan) return;
      const progress = clamp(time / this.duration);
      const point = this.journeyPoint(progress, feature);
      const previous = this.journeyPrev || point;

      this.drawJourneyGround(previous, point, feature, progress, stepIndex);
      this.drawJourneyAtmosphere(point, feature, progress, stepIndex);
      if (Math.abs(point.x - previous.x) > 0.08 || Math.abs(point.y - previous.y) > 0.08) {
        this.drawJourneyThread(previous, point, feature, progress, stepIndex);
      }

      const sectionChanged = feature.sectionIndex !== this.lastSection;
      if (sectionChanged) {
        const ctx = this.detailCtx;
        const veilWidth = Math.max(32, w * 0.07);
        const veil = ctx.createLinearGradient(point.x - veilWidth, 0, point.x + veilWidth, 0);
        veil.addColorStop(0, this.paintColor(feature, progress, feature.sectionIndex, 0, 0.04));
        veil.addColorStop(0.5, this.paintColor(feature, progress, feature.sectionIndex, this.palette.dark ? 0.080 : 0.060, 0.10, 0.48));
        veil.addColorStop(1, this.paintColor(feature, progress, feature.sectionIndex, 0, 0.04));
        ctx.save();
        ctx.globalCompositeOperation = this.palette.dark ? 'screen' : 'multiply';
        ctx.fillStyle = veil;
        ctx.fillRect(point.x - veilWidth, 0, veilWidth * 2, h);
        ctx.restore();
        this.drawJourneyLandmark(point, feature, progress, stepIndex, true);
        this.journeyLastSubjectX = point.x;
        this.lastSection = feature.sectionIndex;
      }

      const subjectGap = w * clamp(0.065 - this.complexity * 0.012, 0.035, 0.060);
      const subjectSignal = feature.onset * 0.58 + feature.energy * 0.25 + feature.mid * 0.17;
      if (point.x - this.journeyLastSubjectX > subjectGap && subjectSignal > 0.39 && randAt(this.seed, stepIndex, 1330) < 0.24 + subjectSignal * 0.34) {
        this.drawJourneyLandmark(point, feature, progress, stepIndex, false);
        this.journeyLastSubjectX = point.x;
      }

      const cloudGap = w * 0.035;
      if (feature.high > 0.46 && point.x - this.journeyLastCloudX > cloudGap && randAt(this.seed, stepIndex, 1331) < feature.high * 0.20) {
        const cloudY = clamp(point.y - h * (0.13 + feature.high * 0.11), h * 0.10, h * 0.52);
        this.ctx = this.detailCtx;
        this.drawCloudStroke(point.x + (randAt(this.seed, stepIndex, 1332) - 0.5) * 20, cloudY, 18 + feature.high * 46, feature, progress, stepIndex + 1900);
        this.journeyLastCloudX = point.x;
      }

      const birdGap = w * 0.055;
      if (feature.onset > 0.52 && feature.high > 0.48 && point.x - this.journeyLastBirdX > birdGap && randAt(this.seed, stepIndex, 1333) < 0.22 + feature.onset * 0.16) {
        this.ctx = this.detailCtx;
        this.drawBirdMark(point.x, clamp(point.y - h * 0.17, h * 0.10, h * 0.55), 3 + feature.high * 5, feature, progress);
        this.journeyLastBirdX = point.x;
      }

      this.journeyPrev = { x: point.x, y: point.y, angle: point.angle, anchor: point.anchor };
      this.composePainterly();
    };
  }

  function installCitadelSculpture() {
    const proto = Sculpture3D.prototype;
    if (proto.__citadelInstalled) return;
    proto.__citadelInstalled = true;

    const originalConfigure = proto.configure;
    const originalCreateInstances = proto.createInstances;
    const originalApplyPalette = proto.applyPalette;
    const originalReset = proto.reset;
    const originalGenerate = proto.generate;
    const originalGenerateSanctuary = proto.generateSanctuary;

    proto.configure = function configureWithCitadel(config) {
      originalConfigure.call(this, config);
      if (this.styleKey === 'citadel') {
        this.totalLength = clamp(this.duration * 0.44, 58, 286);
        this.citadelLastRelicStep = -999;
        this.citadelLastLanternStep = -999;
        this.citadelLastTerraceStep = -999;
      }
    };

    proto.createInstances = function createInstancesWithCitadel() {
      originalCreateInstances.call(this);
      if (this.styleKey === 'citadel') {
        for (const mesh of [this.trunk, this.branches, this.nodes, this.sparks]) if (mesh) mesh.visible = false;

          break;
        }
      }
      const nextLocal = clamp((next - na.progress) / Math.max(0.0001, nb.progress - na.progress));
      const nextY = clamp(lerp(na.y, nb.y, smoothstep(0, 1, nextLocal)) + musicalLift + Math.sin(next * TAU * 1.25 + (na.phase || 0)) * 0.02, 0.10, 0.91) * this.height;
      const y = yNorm * this.height;
      return { x, y, angle: Math.atan2(nextY - y, nextX - x), anchor: b, previousAnchor: a };
    };

    proto.drawJourneyGround = function drawJourneyGround(previous, current, feature, progress, stepIndex) {
      const ctx = this.terrainCtx;
      this.ctx = ctx;
      const groundOffset = this.height * (0.085 + feature.low * 0.065);
      const a = { x: previous.x, y: previous.y + groundOffset };
      const b = { x: current.x, y: current.y + groundOffset };
      const width = this.height * (0.035 + feature.low * 0.065 + feature.energy * 0.018);
      ctx.save();
      ctx.lineCap = 'round';
      ctx.globalCompositeOperation = this.palette.dark ? 'screen' : 'multiply';
      this.drawBristleStroke({
        x1: a.x, y1: a.y, x2: b.x, y2: b.y,
        width,
        curve: (randAt(this.seed, stepIndex, 1321) - 0.5) * width * 0.28,
        feature,
        progress,
        sectionIndex: feature.sectionIndex,
        alpha: this.palette.dark ? 0.12 : 0.10,
        stepIndex,
        channel: 1322,
      });
      ctx.restore();
      this.ctx = this.detailCtx;
    };

    proto.drawJourneyThread = function drawJourneyThread(previous, current, feature, progress, stepIndex) {
      const ctx = this.detailCtx;
      this.ctx = ctx;
      const width = 3.2 + feature.energy * 7.6 + feature.low * 3.8;
      ctx.save();
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      ctx.globalCompositeOperation = this.palette.dark ? 'screen' : 'multiply';
      this.drawBristleStroke({
        x1: previous.x,
        y1: previous.y,
        x2: current.x,
        y2: current.y,
        width,
        curve: (randAt(this.seed, stepIndex, 1323) - 0.5) * (5 + feature.mid * 10),
        feature,
        progress,
        sectionIndex: feature.sectionIndex,
        alpha: this.palette.dark ? 0.33 : 0.28,
        stepIndex,
        channel: 1324,
      });
      ctx.restore();
    };

    proto.drawJourneyAtmosphere = function drawJourneyAtmosphere(point, feature, progress, stepIndex) {
      const ctx = this.detailCtx;
      this.ctx = ctx;
      const radius = this.height * (0.048 + feature.energy * 0.080 + feature.high * 0.030);
      const gradient = ctx.createRadialGradient(point.x, point.y, 0, point.x, point.y, radius);
      gradient.addColorStop(0, this.paintColor(feature, progress, feature.sectionIndex, this.palette.dark ? 0.060 : 0.050, 0.12, 0.52));
      gradient.addColorStop(1, this.paintColor(feature, progress, feature.sectionIndex, 0, 0.06, 0.52));
      ctx.save();
      ctx.globalCompositeOperation = this.palette.dark ? 'screen' : 'multiply';
      ctx.fillStyle = gradient;
      ctx.fillRect(point.x - radius, point.y - radius, radius * 2, radius * 2);
      ctx.restore();

      const accentGap = Math.max(9, Math.round(15 - this.complexity * 4));
      if (stepIndex % accentGap === 0) {
        this.drawKnifeMark(
          point.x + Math.cos(point.angle) * 10,
          point.y + Math.sin(point.angle) * 10,
          point.angle,
          12 + feature.low * 34,
          2.4 + feature.energy * 8,
          feature,
          progress,
          feature.sectionIndex,
          stepIndex,
          1325,
        );
      }
    };

    proto.drawJourneyLandmark = function drawJourneyLandmark(point, feature, progress, stepIndex, force = false) {
      const anchor = point.anchor || { motif: 0, side: 1, openness: 0.5, scale: 1 };
      const normalX = -Math.sin(point.angle);
      const normalY = Math.cos(point.angle);
      const direction = anchor.side || 1;
      const lateral = direction * (24 + this.height * (0.035 + (anchor.openness || 0.5) * 0.045));
      const x = point.x + normalX * lateral;
      const y = point.y + normalY * lateral * 0.20;
      const metrics = this.painterlyMetrics(feature, progress);
      metrics.groundY = clamp(y + this.height * (0.10 + feature.low * 0.055), this.height * 0.38, this.height * 0.93);
      metrics.midY = clamp(y + this.height * 0.025, this.height * 0.20, metrics.groundY - 10);
      metrics.farY = clamp(y - this.height * 0.055, this.height * 0.12, metrics.midY - 8);
      metrics.waterY = clamp(metrics.groundY + this.height * 0.04, metrics.groundY, this.height * 0.94);
      metrics.layout = { ...metrics.layout, motif: anchor.motif, subjectScale: (anchor.scale || 1) * (0.78 + feature.energy * 0.32) };

      this.ctx = this.detailCtx;
      const motif = anchor.motif;
      if (motif === 0) {
        this.drawMountainLandmark(x, metrics, feature, progress, stepIndex + 1400);
      } else if (motif === 1) {
        this.drawGroveLandmark(x, metrics, feature, progress, stepIndex + 1500);
      } else if (motif === 2) {

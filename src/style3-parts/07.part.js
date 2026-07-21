    proto.createCitadelTerrace = function createCitadelTerrace(feature, layout, progress, stepIndex) {
      const flatTangent = this.tangent.clone();
      flatTangent.y = 0;
      if (flatTangent.lengthSq() < 1e-6) flatTangent.set(1, 0, 0);
      flatTangent.normalize();
      const side = new THREE.Vector3(-flatTangent.z, 0, flatTangent.x);
      const sideSign = randAt(this.seed, stepIndex, 1420) < 0.5 ? -1 : 1;
      const group = new THREE.Group();
      const color = this.colorFor(feature, progress, -0.14).clone();
      const bright = this.colorFor(feature, progress, 0.16).clone();
      const material = new THREE.MeshStandardMaterial({ color, roughness: 0.62, metalness: 0.08 });
      const glowMaterial = new THREE.MeshStandardMaterial({ color: bright, roughness: 0.24, metalness: 0.28, emissive: bright.clone().multiplyScalar(this.palette.dark ? 0.16 : 0.04), transparent: true, opacity: 0.76 });
      const base = this.point.clone().addScaledVector(side, sideSign * layout.width * 0.78);
      for (let i = 0; i < 4; i++) {
        const step = new THREE.Mesh(new THREE.BoxGeometry(0.7 + i * 0.22, 0.12, 0.50 + i * 0.18), material);
        step.position.copy(base).addScaledVector(side, sideSign * i * 0.18).add(new THREE.Vector3(0, i * 0.11, 0));
        step.rotation.y = Math.atan2(flatTangent.x, flatTangent.z);
        group.add(step);
      }
      const marker = new THREE.Mesh(new THREE.TorusGeometry(0.32 + feature.high * 0.20, 0.025, 7, 36), glowMaterial);
      marker.position.copy(base).add(new THREE.Vector3(0, 1.1 + feature.mid * 1.4, 0));
      marker.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), flatTangent);
      group.add(marker);
      this.root.add(group);
      this.sectionMeshes.push(group);
      this.expandBounds(base.clone().add(new THREE.Vector3(1.6, 3.2, 1.6)));
      this.expandBounds(base.clone().sub(new THREE.Vector3(1.6, 0.4, 1.6)));
    };

    proto.generateCitadel = function generateCitadel(time, feature, stepIndex) {
      const previousSection = this.lastSection;
      originalGenerateSanctuary.call(this, time, feature, stepIndex);
      const progress = clamp(time / this.duration);
      const layout = this.sanctuaryLayouts[feature.sectionIndex] || {
        width: 5.5, height: 6, motif: 0, phase: 0, centerProgress: progress,
        colorFeature: feature,
      };

      if (feature.sectionIndex !== previousSection) {
        this.createCitadelChapter(feature, layout, progress, stepIndex);
        this.citadelLastRelicStep = stepIndex - 20;
        this.citadelLastLanternStep = stepIndex - 20;
        this.citadelLastTerraceStep = stepIndex - 20;
      }

      const relicGap = Math.max(13, Math.round(24 - this.complexity * 5));
      const relicSignal = feature.onset * 0.54 + feature.mid * 0.24 + feature.high * 0.22;
      if (stepIndex - this.citadelLastRelicStep >= relicGap && relicSignal > 0.42 && randAt(this.seed, stepIndex, 1430) < 0.20 + relicSignal * 0.30) {
        this.createCitadelRelic(feature, layout, progress, stepIndex);
        this.citadelLastRelicStep = stepIndex;
      }

      const lanternGap = Math.max(8, Math.round(16 - this.complexity * 4));
      const lanternSignal = feature.high * 0.62 + feature.centroid * 0.20 + feature.onset * 0.18;
      if (stepIndex - this.citadelLastLanternStep >= lanternGap && lanternSignal > 0.46 && randAt(this.seed, stepIndex, 1431) < 0.18 + lanternSignal * 0.28) {
        this.createCitadelLantern(feature, layout, progress, stepIndex);
        this.citadelLastLanternStep = stepIndex;
      }

      const terraceGap = Math.max(21, Math.round(34 - this.complexity * 6));
      if (stepIndex - this.citadelLastTerraceStep >= terraceGap && feature.low > 0.36 && randAt(this.seed, stepIndex, 1432) < 0.24 + feature.low * 0.20) {
        this.createCitadelTerrace(feature, layout, progress, stepIndex);
        this.citadelLastTerraceStep = stepIndex;
      }
    };
  }
}

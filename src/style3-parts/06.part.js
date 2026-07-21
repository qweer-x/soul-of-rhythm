      this.expandBounds(this.point.clone().addScaledVector(side, width * 0.82).add(new THREE.Vector3(0, height * 1.15, 0)));
      this.expandBounds(this.point.clone().addScaledVector(side, -width * 0.82).add(new THREE.Vector3(0, -0.8, 0)));
    };

    proto.createCitadelRelic = function createCitadelRelic(feature, layout, progress, stepIndex) {
      const flatTangent = this.tangent.clone();
      flatTangent.y = 0;
      if (flatTangent.lengthSq() < 1e-6) flatTangent.set(1, 0, 0);
      flatTangent.normalize();
      const side = new THREE.Vector3(-flatTangent.z, 0, flatTangent.x);
      const sideSign = randAt(this.seed, stepIndex, 1400) < 0.5 ? -1 : 1;
      const group = new THREE.Group();
      const base = this.point.clone().addScaledVector(side, sideSign * layout.width * (0.58 + randAt(this.seed, stepIndex, 1401) * 0.30));
      const color = this.colorFor(feature, progress, 0.05).clone();
      const bright = this.colorFor(feature, progress, 0.24).clone();
      const material = new THREE.MeshStandardMaterial({ color, roughness: 0.38, metalness: 0.20, emissive: color.clone().multiplyScalar(this.palette.dark ? 0.06 : 0.015) });
      const glowMaterial = new THREE.MeshStandardMaterial({ color: bright, roughness: 0.18, metalness: 0.34, emissive: bright.clone().multiplyScalar(this.palette.dark ? 0.24 : 0.06), transparent: true, opacity: 0.86 });
      const relicHeight = 1.5 + feature.mid * 3.8 + feature.onset * 1.4;

      if (feature.low > feature.high) {
        const slab = new THREE.Mesh(new THREE.BoxGeometry(0.32 + feature.low * 0.22, relicHeight, 0.24 + feature.low * 0.14), material);
        slab.position.copy(base).add(new THREE.Vector3(0, relicHeight * 0.5, 0));
        slab.rotation.y = Math.atan2(flatTangent.x, flatTangent.z) + sideSign * 0.20;
        group.add(slab);
        const sigil = new THREE.Mesh(new THREE.TorusGeometry(0.18 + feature.high * 0.18, 0.025, 7, 30), glowMaterial);
        sigil.position.copy(slab.position).add(new THREE.Vector3(0, relicHeight * 0.13, 0.16));
        sigil.rotation.y = slab.rotation.y;
        group.add(sigil);
      } else {
        const count = 3 + Math.floor(feature.high * 3);
        for (let i = 0; i < count; i++) {
          const shard = new THREE.Mesh(new THREE.OctahedronGeometry(0.12 + feature.high * 0.16, 0), i === count - 1 ? glowMaterial : material);
          shard.position.copy(base)
            .addScaledVector(side, (randAt(this.seed, stepIndex * 7 + i, 1402) - 0.5) * 1.2)
            .addScaledVector(flatTangent, (randAt(this.seed, stepIndex * 11 + i, 1403) - 0.5) * 1.0)
            .add(new THREE.Vector3(0, 0.7 + i * (0.38 + feature.high * 0.18), 0));
          shard.rotation.set(i * 0.7, layout.phase + i, i * 0.4);
          group.add(shard);
        }
      }

      this.root.add(group);
      this.sectionMeshes.push(group);
      this.expandBounds(base.clone().add(new THREE.Vector3(1.2, relicHeight + 1.5, 1.2)));
      this.expandBounds(base.clone().sub(new THREE.Vector3(1.2, 0.5, 1.2)));
    };

    proto.createCitadelLantern = function createCitadelLantern(feature, layout, progress, stepIndex) {
      const flatTangent = this.tangent.clone();
      flatTangent.y = 0;
      if (flatTangent.lengthSq() < 1e-6) flatTangent.set(1, 0, 0);
      flatTangent.normalize();
      const side = new THREE.Vector3(-flatTangent.z, 0, flatTangent.x);
      const group = new THREE.Group();
      const bright = this.colorFor(feature, progress, 0.26).clone();
      const dark = this.colorFor(feature, progress, -0.10).clone();
      const wireMaterial = new THREE.MeshStandardMaterial({ color: dark, roughness: 0.48, metalness: 0.24 });
      const glowMaterial = new THREE.MeshStandardMaterial({ color: bright, roughness: 0.16, metalness: 0.30, emissive: bright.clone().multiplyScalar(this.palette.dark ? 0.30 : 0.07), transparent: true, opacity: 0.90 });
      const sideSign = randAt(this.seed, stepIndex, 1410) < 0.5 ? -1 : 1;
      const base = this.point.clone().addScaledVector(side, sideSign * layout.width * (0.30 + randAt(this.seed, stepIndex, 1411) * 0.36));
      const topY = layout.height * (0.52 + feature.high * 0.44);
      const stem = new THREE.Mesh(new THREE.CylinderGeometry(0.018, 0.026, topY, 6), wireMaterial);
      stem.position.copy(base).add(new THREE.Vector3(0, topY * 0.5, 0));
      group.add(stem);
      const bellCount = feature.high > 0.74 ? 3 : 2;
      for (let i = 0; i < bellCount; i++) {
        const bell = new THREE.Mesh(new THREE.IcosahedronGeometry(0.12 + feature.high * 0.08, 1), glowMaterial);
        bell.position.copy(base)
          .addScaledVector(flatTangent, (i - (bellCount - 1) * 0.5) * 0.42)
          .add(new THREE.Vector3(0, topY - i * 0.22, 0));
        group.add(bell);
      }
      this.root.add(group);
      this.sectionMeshes.push(group);
      this.expandBounds(base.clone().add(new THREE.Vector3(1, topY + 1, 1)));
    };


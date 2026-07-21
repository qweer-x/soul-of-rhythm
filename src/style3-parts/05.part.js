        for (const mesh of [this.floors, this.foundations, this.columns, this.beams, this.panels, this.ornaments]) if (mesh) mesh.visible = true;
      }
    };

    proto.applyPalette = function applyPaletteWithCitadel() {
      originalApplyPalette.call(this);
      if (this.styleKey === 'citadel') {
        this.grid.position.y = -2.8;
        this.grid.material.opacity = this.palette.dark ? 0.045 : 0.065;
        this.keyLight.intensity = 2.20;
        this.rimLight.intensity = 1.90;
        this.tipLight.intensity = 2.55;
      }
    };

    proto.reset = function resetWithCitadel() {
      originalReset.call(this);
      this.citadelLastRelicStep = -999;
      this.citadelLastLanternStep = -999;
      this.citadelLastTerraceStep = -999;
    };

    proto.generate = function generateWithCitadel(time, feature, stepIndex) {
      if (this.styleKey === 'citadel') this.generateCitadel(time, feature, stepIndex);
      else originalGenerate.call(this, time, feature, stepIndex);
    };

    proto.createCitadelChapter = function createCitadelChapter(feature, layout, progress, stepIndex) {
      const flatTangent = this.tangent.clone();
      flatTangent.y = 0;
      if (flatTangent.lengthSq() < 1e-6) flatTangent.set(1, 0, 0);
      flatTangent.normalize();
      const side = new THREE.Vector3(-flatTangent.z, 0, flatTangent.x);
      const group = new THREE.Group();
      group.position.copy(this.point);
      this.orientSanctuaryGroup(group, flatTangent);

      const stable = this.sanctuaryFeature(layout, feature, 0.08);
      const stone = this.colorFor(stable, layout.centerProgress, -0.08).clone();
      const shadow = this.colorFor(stable, layout.centerProgress, -0.22).clone();
      const bright = this.colorFor(feature, progress, 0.22).clone();
      const stoneMaterial = new THREE.MeshStandardMaterial({ color: stone, roughness: 0.52, metalness: 0.12, emissive: stone.clone().multiplyScalar(this.palette.dark ? 0.04 : 0.01) });
      const shadowMaterial = new THREE.MeshStandardMaterial({ color: shadow, roughness: 0.72, metalness: 0.04 });
      const glowMaterial = new THREE.MeshStandardMaterial({ color: bright, roughness: 0.18, metalness: 0.32, emissive: bright.clone().multiplyScalar(this.palette.dark ? 0.26 : 0.06), transparent: true, opacity: 0.88 });
      const glassMaterial = new THREE.MeshPhysicalMaterial({ color: bright.clone().offsetHSL(0, -0.12, 0.08), roughness: 0.14, metalness: 0.08, transmission: 0.18, transparent: true, opacity: this.palette.dark ? 0.42 : 0.30, side: THREE.DoubleSide });
      const width = layout.width * (1.02 + feature.low * 0.18);
      const height = layout.height * (1.04 + feature.mid * 0.22);

      for (let i = 0; i < 3; i++) {
        const terrace = new THREE.Mesh(new THREE.BoxGeometry(1, 1, 1), i === 0 ? shadowMaterial : stoneMaterial);
        terrace.position.set(0, -0.28 + i * 0.17, -0.16 * i);
        terrace.scale.set(width * (1.18 - i * 0.14), 0.16 + feature.low * 0.07, width * (0.82 - i * 0.08));
        group.add(terrace);
      }

      const towerCount = layout.motif % 2 === 0 ? 2 : 3;
      for (let i = 0; i < towerCount; i++) {
        const lateral = towerCount === 2 ? (i === 0 ? -1 : 1) : i - 1;
        const towerHeight = height * (0.60 + i * 0.10 + feature.mid * 0.18);
        const tower = new THREE.Mesh(new THREE.CylinderGeometry(0.15 + feature.low * 0.05, 0.22 + feature.low * 0.07, towerHeight, 10), stoneMaterial);
        tower.position.set(lateral * width * 0.40, towerHeight * 0.5, width * (0.08 + (i % 2) * 0.10));
        group.add(tower);
        const crown = new THREE.Mesh(new THREE.OctahedronGeometry(0.18 + feature.high * 0.10, 0), glowMaterial);
        crown.position.copy(tower.position).add(new THREE.Vector3(0, towerHeight * 0.5 + 0.12, 0));
        crown.rotation.set(layout.phase + i, layout.phase * 0.5, i * 0.7);
        group.add(crown);
      }

      if (layout.motif === 0 || layout.motif === 2) {
        for (let i = 0; i < 3; i++) {
          const ring = new THREE.Mesh(new THREE.TorusGeometry(width * (0.24 + i * 0.075), 0.035 + i * 0.010, 8, 54), i === 2 ? glowMaterial : glassMaterial);
          ring.position.set(0, height * (0.52 + i * 0.12), width * (0.04 - i * 0.06));
          ring.rotation.x = Math.PI / 2;
          group.add(ring);
        }
      } else {
        const pool = new THREE.Mesh(new THREE.CylinderGeometry(width * 0.28, width * 0.32, 0.07, 24), glassMaterial);
        pool.position.set(0, 0.12, -width * 0.16);
        group.add(pool);
        const core = new THREE.Mesh(new THREE.IcosahedronGeometry(0.28 + feature.high * 0.18, 1), glowMaterial);
        core.position.set(0, height * 0.38, -width * 0.16);
        group.add(core);
      }

      const light = new THREE.PointLight(bright, this.palette.dark ? 1.8 : 0.85, 18, 2);
      light.position.set(0, height * 0.54, width * 0.20);
      group.add(light);

      this.root.add(group);
      this.sectionMeshes.push(group);

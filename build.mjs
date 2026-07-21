import { build } from 'esbuild';
import fs from 'node:fs/promises';
import path from 'node:path';

const root = path.resolve(process.cwd());
const mainPath = path.join(root, 'src/main.js');
const style3Path = path.join(root, 'src/style3.js');
const style3PartsDir = path.join(root, 'src/style3-parts');
const initMarker = 'init().catch(showError);';
const exposeSource = `
globalThis.__SONIC_CANVAS__ = {
  THREE, TAU, clamp, lerp, smoothstep, randAt, valueNoise1D, sampleAnalysis,
  PALETTES, PAINTING_STYLES, SCULPTURE_STYLES, paletteHsl,
  CanvasPainting, Sculpture3D, state, painting, sculpture, els,
  updateModeUI, updatePaintingStyleUI, updateSculptureStyleUI,
  configureVisuals, rebuildTo, init, showError,
};
`;

const exposeMainPlugin = {
  name: 'expose-sonic-canvas-runtime',
  setup(builder) {
    builder.onLoad({ filter: /[\\/]src[\\/]main\.js$/ }, async (args) => {
      const source = await fs.readFile(args.path, 'utf8');
      if (!source.includes(initMarker)) {
        throw new Error('Cannot expose runtime: init marker was not found in src/main.js');
      }
      return {
        contents: source.replace(initMarker, exposeSource),
        loader: 'js',
        resolveDir: path.dirname(args.path),
      };
    });

    builder.onLoad({ filter: /[\/]src[\/]style3\.js$/ }, async () => {
      const partNames = (await fs.readdir(style3PartsDir))
        .filter((name) => name.endsWith('.part.js'))
        .sort();
      if (!partNames.length) throw new Error('No style 3 source fragments were found');
      const parts = await Promise.all(partNames.map((name) => fs.readFile(path.join(style3PartsDir, name), 'utf8')));
      return {
        contents: parts.join('\n'),
        loader: 'js',
        resolveDir: path.dirname(style3Path),
      };
    });
  },
};

const result = await build({
  entryPoints: [path.join(root, 'src/entry.js')],
  bundle: true,
  minify: true,
  format: 'iife',
  target: ['es2020'],
  write: false,
  sourcemap: false,
  legalComments: 'none',
  plugins: [exposeMainPlugin],
});
const js = result.outputFiles[0].text;
const css = await fs.readFile(path.join(root, 'src/styles.css'), 'utf8');
const template = await fs.readFile(path.join(root, 'src/index.template.html'), 'utf8');
const html = template.replace('/*__CSS__*/', () => css).replace('/*__JS__*/', () => js);
await fs.writeFile(path.join(root, '音乐画布_2D+3D原型.html'), html);
console.log(`built ${Buffer.byteLength(html)} bytes`);

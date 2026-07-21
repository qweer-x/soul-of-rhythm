import { build } from 'esbuild';
import fs from 'node:fs/promises';
import path from 'node:path';

const root = path.resolve(process.cwd());
const result = await build({
  entryPoints: [path.join(root, 'src/main.js')],
  bundle: true,
  minify: true,
  format: 'iife',
  target: ['es2020'],
  write: false,
  sourcemap: false,
  legalComments: 'none',
});
const js = result.outputFiles[0].text;
const css = await fs.readFile(path.join(root, 'src/styles.css'), 'utf8');
const template = await fs.readFile(path.join(root, 'src/index.template.html'), 'utf8');
const html = template.replace('/*__CSS__*/', () => css).replace('/*__JS__*/', () => js);
await fs.writeFile(path.join(root, '音乐画布_2D+3D原型.html'), html);
console.log(`built ${Buffer.byteLength(html)} bytes`);

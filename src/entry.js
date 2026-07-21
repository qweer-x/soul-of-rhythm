import './main.js';
import { installStyle3 } from './style3.js';

const runtime = globalThis.__SONIC_CANVAS__;
if (!runtime) throw new Error('Sonic Canvas runtime was not exposed by the build step');
installStyle3(runtime);
runtime.init().catch(runtime.showError);

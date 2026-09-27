import { build } from 'esbuild';
import { fileURLToPath } from 'node:url';
import path from 'node:path';
const root = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
await build({absWorkingDir: root, entryPoints: ['vendor/liveavatar-entry.js'], bundle: true,
  format: 'iife', globalName: 'LiveAvatarSDK', outfile: 'js/vendor/liveavatar.js',
  minify: true, legalComments: 'linked', platform: 'browser'});

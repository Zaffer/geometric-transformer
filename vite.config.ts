import { defineConfig } from 'vite';

export default defineConfig({
  server: { port: 3000 },
  worker: { format: 'es' },
  optimizeDeps: { include: ['three', 'three/addons/controls/OrbitControls.js'] },
});

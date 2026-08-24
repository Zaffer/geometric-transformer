import { defineConfig } from 'vite';

export default defineConfig({
  server: { port: 3000 },
  optimizeDeps: { include: ['three/webgpu', 'three/addons/controls/OrbitControls.js'] },
});

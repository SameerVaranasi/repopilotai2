import {defineConfig} from 'vite';
export default defineConfig({server:{proxy:{'/api':{target:process.env.REPOPILOT_BACKEND_URL || 'http://127.0.0.1:8000',changeOrigin:true,rewrite:p=>p.replace(/^\/api/,'')}}}});

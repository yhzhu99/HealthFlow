import { createApp } from 'vue'
import '@fontsource/instrument-sans/400.css'
import '@fontsource/instrument-sans/500.css'
import '@fontsource/instrument-sans/600.css'
import '@fontsource/instrument-sans/700.css'
import 'katex/dist/katex.min.css'
import 'markdown-it-texmath/css/texmath.css'

import './style.css'
import App from './App.vue'
import router from './router'

createApp(App).use(router).mount('#app')

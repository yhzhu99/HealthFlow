import { createApp } from 'vue'
import '@fontsource/manrope/400.css'
import '@fontsource/manrope/500.css'
import '@fontsource/manrope/600.css'
import '@fontsource/instrument-serif/400.css'

import './style.css'
import App from './App.vue'
import router from './router'

createApp(App).use(router).mount('#app')

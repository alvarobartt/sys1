import DefaultTheme from 'vitepress/theme';
import ModelPublisher from './ModelPublisher.vue';
import '../mono.css';

export default {
  extends: DefaultTheme,
  enhanceApp({ app }) {
    app.component('ModelPublisher', ModelPublisher);
  }
};

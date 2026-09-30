const base = process.env.DOCS_BASE || '/';

export default {
  base,
  title: 'sys1',
  description: 'Blazing fast, self-hosted structured decisions for open-weight models with a TypeSafe AI compatible API, written in Rust.',
  cleanUrls: true,
  srcExclude: ['README.md'],
  head: [
    ['link', { rel: 'icon', type: 'image/png', href: `${base}favicon.png` }],
    ['link', { rel: 'apple-touch-icon', href: `${base}apple-touch-icon.png` }],
    ['meta', { name: 'theme-color', content: '#ffffff' }]
  ],
  themeConfig: {
    logo: false,
    siteTitle: 'sys1',
    sidebar: [
      { text: 'Overview', link: '/' },
      {
        text: 'Get started',
        link: '/getting-started/',
        items: [
          { text: 'CPU', link: '/getting-started/cpu' },
          { text: 'Metal', link: '/getting-started/metal' },
          { text: 'CUDA', link: '/getting-started/cuda' },
          { text: 'Docker', link: '/getting-started/docker' }
        ]
      },
      {
        text: 'Advanced',
        items: [
          { text: 'Batching', link: '/advanced/batching' }
        ]
      },
      {
        text: 'Models',
        link: '/models/',
        items: [
          { text: 'Laya', link: '/models/laya' },
          { text: 'Laya Multilingual', link: '/models/laya-multilingual' },
          { text: 'Laya Typed Decisions', link: '/models/laya-typed-decisions' }
        ]
      },
      { text: 'CLI', link: '/cli' },
      {
        text: 'API',
        link: '/api',
        collapsed: true,
        items: [
          { text: 'Health', link: '/api#health' },
          { text: 'List models', link: '/api#list-models' },
          { text: 'System One', link: '/api#system-one' },
          { text: 'Metrics', link: '/api#metrics' }
        ]
      },
      { text: 'Changelog', link: '/changelog' }
    ],
    socialLinks: [
      { icon: 'github', link: 'https://github.com/alvarobartt/sys1' }
    ],
    search: { provider: 'local' },
    outline: { level: [2, 3] },
    footer: { message: 'Released under the Apache-2.0 license.' }
  }
}

export default {
  title: 'Snake Math',
  description: 'Interactive mathematical concepts powered by Python in your browser',
  head: [
    ['script', { 
      type: 'text/javascript', 
      src: 'https://pyscript.net/releases/2024.1.1/core.js' 
    }]
  ],
  base: '/snake-math/',
  themeConfig: {
    nav: [
      { text: 'Home', link: '/' },
      { text: 'Basics', link: '/basics/' }
    ],
    sidebar: {
      '/': [
        {
          text: 'Getting Started',
          items: [
            { text: 'Introduction', link: '/' },
            { text: 'Mathematical Foundations', link: '/basics/' }
          ]
        }
      ]
    }
  }
}
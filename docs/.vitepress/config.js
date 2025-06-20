export default {
  title: 'Snake Math',
  description: 'Interactive mathematical concepts powered by Python in your browser',
  head: [
    // PyScript 2025.5.1 CSS
    ['link', { 
      rel: 'stylesheet', 
      href: 'https://pyscript.net/releases/2025.5.1/core.css' 
    }],
    // PyScript 2025.5.1 Core JS  
    ['script', { 
      type: 'module',
      src: 'https://pyscript.net/releases/2025.5.1/core.js' 
    }],
    // Optional: Add some custom styling for PyScript output
    ['style', {}, `
      .py-terminal {
        background: #f8f9fa;
        border-left: 4px solid #4CAF50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
        font-family: 'Courier New', monospace;
      }
      
      .py-terminal:before {
        content: "🐍 Python Output:";
        display: block;
        font-weight: bold;
        color: #2E7D32;
        margin-bottom: 0.5rem;
        font-family: system-ui, sans-serif;
      }
    `]
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
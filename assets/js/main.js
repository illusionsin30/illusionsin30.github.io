document.addEventListener("DOMContentLoaded", function() {
  // 1. TOC 逻辑 (保持不变)
  const tocContainer = document.getElementById('toc');
  const content = document.querySelector('.post-body');

  if (tocContainer && content) {
    // ... (保持你之前的 TOC 代码不变) ...
    // 如果没有 TOC 代码，请把上次回答的 JS 代码复制在这里
    const headings = content.querySelectorAll('h2, h3');
    if (headings.length > 0) {
      const ul = document.createElement('ul');
      headings.forEach((heading, index) => {
        const id = heading.id || `heading-${index}`;
        heading.id = id;
        const li = document.createElement('li');
        const a = document.createElement('a');
        a.href = `#${id}`;
        a.textContent = heading.textContent;
        if(heading.tagName === 'H3') li.style.marginLeft = '15px';
        li.appendChild(a);
        ul.appendChild(li);
      });
      tocContainer.appendChild(ul);
    } else {
      document.querySelector('#toc-container').style.display = 'none';
    }
  }

  // 2. 初始化主题图标状态
  const currentTheme = document.documentElement.getAttribute('data-theme');
  updateThemeUI(currentTheme === 'dark');
});

// === Theme Toggle Logic ===
function toggleTheme() {
  const html = document.documentElement;
  const current = html.getAttribute('data-theme');
  const isDark = current === 'dark';
  
  if (isDark) {
    html.removeAttribute('data-theme');
    localStorage.setItem('theme', 'light');
    updateThemeUI(false);
  } else {
    html.setAttribute('data-theme', 'dark');
    localStorage.setItem('theme', 'dark');
    updateThemeUI(true);
  }
}

function updateThemeUI(isDark) {
  const icon = document.querySelector('#theme-icon i');
  const text = document.querySelector('#theme-text');
  
  if (icon && text) {
    if (isDark) {
      icon.className = 'fas fa-sun'; // 变太阳
      text.textContent = 'Light Mode';
    } else {
      icon.className = 'fas fa-moon'; // 变月亮
      text.textContent = 'Dark Mode';
    }
  }
}
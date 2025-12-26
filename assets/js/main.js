document.addEventListener("DOMContentLoaded", function() {
  // 1. 获取容器
  const tocContainer = document.getElementById('toc');
  // 尝试匹配不同的正文容器类名（适配 Wiki 或 Post）
  const content = document.querySelector('.post-body') || document.querySelector('.main-content');

  if (tocContainer && content) {
    // 【关键修改点】：增加 h1 和 h4 的抓取
    const headings = content.querySelectorAll('h1, h2, h3, h4');
    
    if (headings.length > 0) {
      const ul = document.createElement('ul');
      
      headings.forEach((heading, index) => {
        // 生成唯一 ID（如果标题没有 ID 的话）
        const id = heading.id || `heading-${index}`;
        heading.id = id;

        const li = document.createElement('li');
        // 根据标题级别添加类名，例如 toc-h1, toc-h2 等
        li.classList.add(`toc-li`, `toc-${heading.tagName.toLowerCase()}`);

        const a = document.createElement('a');
        a.href = `#${id}`;
        a.textContent = heading.textContent;
        
        // 当点击目录项时，如果是手机端，可以自动关闭侧边栏（可选）
        a.addEventListener('click', () => {
           if(window.innerWidth < 850) {
             // 如果你有侧边栏切换逻辑，可以在这里调用
           }
        });

        li.appendChild(a);
        ul.appendChild(li);
      });
      
      tocContainer.appendChild(ul);
    } else {
      // 如果页面没有标题，隐藏整个目录容器
      const container = document.querySelector('#toc-container') || document.querySelector('.post-sidebar-meta');
      if (container) container.style.display = 'none';
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
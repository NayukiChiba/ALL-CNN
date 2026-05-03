# 文档部署指南

## 技术栈

本文档使用 [VitePress](https://vitepress.dev/) 构建，这是一个基于 Vite 和 Vue 的静态站点生成器。

- **Markdown** 编写内容
- **MathJax** 渲染 LaTeX 数学公式
- **VitePress 本地搜索** 提供中文搜索
- **GitHub Actions** 自动部署到 GitHub Pages

---

## 本地预览

### 1. 安装 Node.js

VitePress 是 Node.js 项目。确保安装 Node.js 18+ 版本：

```bash
node --version  # 应 ≥ 18
```

### 2. 安装依赖

```bash
cd docs
npm install
```

### 3. 启动开发服务器

```bash
npm run docs:dev
```

浏览器打开 `http://localhost:5173`，支持热更新——修改 Markdown 文件后页面自动刷新。

### 4. 构建生产版本

```bash
npm run docs:build
```

构建产物在 `docs/.vitepress/dist/` 目录下。

### 5. 预览生产版本

```bash
npm run docs:preview
```

这会启动一个本地静态服务器，模拟线上环境（包括正确的 `base` 路径）。

---

## GitHub Pages 自动部署

### 工作原理

每次推送到 `main` 分支且变更了 `docs/` 目录时，GitHub Actions 自动触发部署：

```
Push to main (docs/** 变更)
    │
    ▼
.github/workflows/deploy-docs.yml
    │
    ├─ 1. Checkout 代码
    ├─ 2. Setup Node.js 22
    ├─ 3. npm ci (安装依赖)
    ├─ 4. npx vitepress build (构建)
    ├─ 5. Upload artifact (上传构建产物)
    └─ 6. Deploy to GitHub Pages
```

### 工作流文件

CI 定义在 [.github/workflows/deploy-docs.yml](https://github.com/NayukiChiba/MNIST-CNN/blob/main/.github/workflows/deploy-docs.yml)：

```yaml
on:
  push:
    branches: [main]
    paths:
      - 'docs/**'
      - '.github/workflows/deploy-docs.yml'
  workflow_dispatch:  # 允许手动触发
```

- `paths` 过滤确保只有文档变更才触发构建，Python 代码变更不会浪费 CI 资源
- `workflow_dispatch` 允许从 Actions 页面手动重新部署

### 首次部署设置

1. 推送包含 `deploy-docs.yml` 的代码到 `main`
2. 前往 GitHub 仓库 **Settings > Pages**
3. 在 **Build and deployment** 中，将 Source 设为 **GitHub Actions**
4. GitHub 会自动检测工作流文件并启动首次部署

### 验证部署

部署完成后：

1. 前往 **Actions** 页签，确认 `Deploy Docs to GitHub Pages` workflow 运行成功
2. 访问 `https://nayukichiba.github.io/MNIST-CNN/`
3. 检查导航、数学公式渲染、搜索功能是否正常

---

## 配置说明

### base 路径

`docs/.vitepress/config.mts` 中的 `base: '/MNIST-CNN/'` 配置决定了站点路径：

- GitHub Pages 仓库站点默认 URL 为 `https://<user>.github.io/<repo>/`
- `base` 必须与仓库名匹配，否则静态资源（JS/CSS/图片）加载失败

**如果使用自定义域名：** 将 `base` 改为 `'/'` 并配置 GitHub Pages 的 Custom domain。

### 搜索

VitePress 使用**本地搜索**（`provider: 'local'`），在构建时生成搜索索引：

- 无需 Algolia API key 或任何第三方服务
- 支持中文分词
- 搜索索引包含在构建产物中

### 数学公式

使用 `markdown-it-mathjax3` 插件，支持两种语法：

```
行内公式：\( \sigma(x) = \frac{1}{1 + e^{-x}} \)

块级公式：
$$
\frac{\partial \mathcal{L}}{\partial z_i} = p_i - \delta_{i,y}
$$
```

---

## 添加新页面

1. 在 `docs/` 下创建 `.md` 文件（如 `docs/math/new-topic.md`）
2. 将页面链接添加到 `docs/.vitepress/config.mts` 的 `sidebar` 配置中

```typescript
sidebar: {
  '/math/': [
    {
      text: '数学原理',
      items: [
        // ... 现有条目 ...
        { text: '新主题', link: '/math/new-topic' },
      ],
    },
  ],
}
```

3. 本地预览确认无误后推送，GitHub Actions 自动重建部署

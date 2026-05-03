import { defineConfig } from 'vitepress'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

export default defineConfig({
  lang: 'zh-CN',
  title: 'MNIST-CNN 文档',
  description: '基于 PyTorch 的 MNIST 手写数字识别 CNN——数学原理与项目架构',
  base: '/MNIST-CNN/',
  cleanUrls: true,
  lastUpdated: true,

  head: [
    ['link', { rel: 'icon', href: '/MNIST-CNN/favicon.ico' }],
  ],

  markdown: {
    math: true,
  },

  vite: {
    server: {
      fs: {
        allow: ['../..'],
      },
    },
    plugins: [
      {
        name: 'serve-visualizations',
        resolveId(id) {
          if (id.startsWith('/visualizations/')) {
            return path.resolve(__dirname, '../..', id.slice(1))
          }
        },
      },
    ],
  },

  themeConfig: {
    search: {
      provider: 'local',
      options: {
        translations: {
          button: { buttonText: '搜索文档' },
          modal: { noResultsText: '未找到相关结果' },
        },
      },
    },

    nav: [
      { text: '首页', link: '/' },
      { text: '数学原理', link: '/math/convolution' },
      { text: '项目架构', link: '/architecture/overview' },
    ],

    sidebar: {
      '/math/': [
        {
          text: '数学原理',
          collapsed: false,
          items: [
            { text: '离散卷积原理', link: '/math/convolution' },
            { text: '各层公式与设计原理', link: '/math/layers' },
            { text: '完整前向传播方程链', link: '/math/forward-pass' },
            { text: '损失函数推导', link: '/math/loss-function' },
            { text: 'Adam 优化器', link: '/math/optimizer' },
          ],
        },
      ],
      '/architecture/': [
        {
          text: '项目架构',
          collapsed: false,
          items: [
            { text: '总体概览', link: '/architecture/overview' },
            { text: '数据管道', link: '/architecture/data-pipeline' },
            { text: '模型设计与工厂模式', link: '/architecture/model' },
            { text: '训练流程', link: '/architecture/training' },
            { text: '评估系统', link: '/architecture/evaluation' },
            { text: '推理系统', link: '/architecture/inference' },
          ],
        },
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/NayukiChiba/MNIST-CNN' },
    ],

    footer: {
      message: '基于 PyTorch 实现 | 遵循 MIT 协议',
    },

    editLink: {
      pattern: 'https://github.com/NayukiChiba/MNIST-CNN/edit/main/docs/:path',
      text: '在 GitHub 上编辑此页',
    },

    lastUpdated: {
      text: '最后更新于',
    },
  },
})

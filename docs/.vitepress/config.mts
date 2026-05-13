import { defineConfig } from 'vitepress'
import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

export default defineConfig({
  lang: 'zh-CN',
  title: 'CNN 文档',
  description: '8 种经典 CNN 架构 x 10 个数据集 | PyTorch 实现 | 数学原理与工程架构',
  base: '/CNN/',
  cleanUrls: true,
  lastUpdated: true,

  head: [
    ['link', { rel: 'icon', href: '/CNN/favicon.ico' }],
  ],

  markdown: {
    math: true,
    config(md) {
      const fallback = md.renderer.rules.fence!
      md.renderer.rules.fence = (tokens, idx, options, env, slf) => {
        const token = tokens[idx]
        if (token.info?.trim() === 'mermaid') {
          return `<pre class="mermaid" style="display:none">${token.content}</pre>`
        }
        return fallback(tokens, idx, options, env, slf)
      }
    },
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
      { text: '快速开始', link: '/guides/quickstart' },
      { text: '模型架构', link: '/models/overview' },
      { text: '数据集', link: '/datasets/overview' },
      { text: '数学原理', link: '/math/convolution' },
      { text: '工程架构', link: '/architecture/overview' },
    ],

    sidebar: {
      '/guides/': [
        {
          text: '使用指南',
          collapsed: false,
          items: [
            { text: '快速开始', link: '/guides/quickstart' },
            { text: '训练指南', link: '/guides/training-guide' },
            { text: '推理指南', link: '/guides/inference-guide' },
            { text: '基准测试指南', link: '/guides/benchmark-guide' },
          ],
        },
        {
          text: '对比参考',
          collapsed: false,
          items: [
            { text: '模型对比总览', link: '/guides/model-comparison' },
            { text: '数据集对比总览', link: '/guides/dataset-comparison' },
          ],
        },
      ],

      '/models/': [
        {
          text: '模型族系',
          collapsed: false,
          items: [
            { text: '总览与对比', link: '/models/overview' },
            { text: 'LeNet-5 (1998)', link: '/models/lenet' },
            { text: 'AlexNet (2012)', link: '/models/alexnet' },
            { text: 'VGG-11/13/16/19 (2015)', link: '/models/vgg' },
            { text: 'NiN — Network In Network (2014)', link: '/models/nin' },
            { text: 'GoogLeNet / Inception v1 (2015)', link: '/models/googlenet' },
          ],
        },
      ],

      '/datasets/': [
        {
          text: '数据集总览',
          collapsed: false,
          items: [
            { text: '总览与对比', link: '/datasets/overview' },
            { text: 'MNIST', link: '/datasets/mnist' },
            { text: 'FashionMNIST', link: '/datasets/fashionmnist' },
            { text: 'EMNIST', link: '/datasets/emnist' },
            { text: 'CIFAR-10', link: '/datasets/cifar10' },
            { text: 'CIFAR-100', link: '/datasets/cifar100' },
            { text: 'SVHN', link: '/datasets/svhn' },
            { text: 'STL-10', link: '/datasets/stl10' },
            { text: 'Caltech-101', link: '/datasets/caltech101' },
            { text: 'GTSRB', link: '/datasets/gtsrb' },
            { text: 'Flowers-102', link: '/datasets/flowers102' },
          ],
        },
      ],

      '/math/': [
        {
          text: '基础数学',
          collapsed: false,
          items: [
            { text: '离散卷积原理', link: '/math/convolution' },
            { text: '各层公式与设计原理', link: '/math/layers' },
            { text: '完整前向传播', link: '/math/forward-pass' },
            { text: 'Softmax 与数值稳定性', link: '/math/softmax' },
            { text: '损失函数推导', link: '/math/loss-function' },
          ],
        },
        {
          text: '初始化与归一化',
          collapsed: false,
          items: [
            { text: 'Kaiming & Xavier 初始化', link: '/math/initialization' },
            { text: 'Batch Normalization', link: '/math/batch-normalization' },
          ],
        },
        {
          text: '正则化',
          collapsed: false,
          items: [
            { text: 'Dropout', link: '/math/dropout' },
            { text: 'L1 / L2 / Weight Decay', link: '/math/regularization' },
          ],
        },
        {
          text: '优化与调度',
          collapsed: false,
          items: [
            { text: 'Adam / AdamW / SGD / RMSprop', link: '/math/optimizer' },
            { text: '学习率调度器', link: '/math/schedulers' },
            { text: '梯度裁剪', link: '/math/gradient-clipping' },
          ],
        },
        {
          text: '分析与计算',
          collapsed: false,
          items: [
            { text: '感受野计算', link: '/math/receptive-field' },
            { text: '参数量计算', link: '/math/parameter-count' },
            { text: 'FLOPs 估算', link: '/math/flops' },
          ],
        },
        {
          text: '数据处理',
          collapsed: false,
          items: [
            { text: '数据增强', link: '/math/augmentations' },
          ],
        },
      ],

      '/architecture/': [
        {
          text: '架构总览',
          collapsed: false,
          items: [
            { text: '总体概览', link: '/architecture/overview' },
            { text: '注册系统 (Registry)', link: '/architecture/registry' },
          ],
        },
        {
          text: '核心模块',
          collapsed: false,
          items: [
            { text: '数据管道', link: '/architecture/data-pipeline' },
            { text: '模型工厂', link: '/architecture/model-factory' },
            { text: '训练流程', link: '/architecture/training' },
            { text: '评估系统', link: '/architecture/evaluation' },
            { text: '推理系统', link: '/architecture/inference' },
          ],
        },
        {
          text: '实验与工具',
          collapsed: false,
          items: [
            { text: '基准测试系统', link: '/architecture/benchmark' },
            { text: 'CLI 系统', link: '/architecture/cli' },
          ],
        },
      ],
    },

    socialLinks: [
      { icon: 'github', link: 'https://github.com/NayukiChiba/CNN' },
    ],

    footer: {
      message: '基于 PyTorch 实现 | 8 种 CNN · 10 个数据集 | 遵循 MIT 协议',
    },

    editLink: {
      pattern: 'https://github.com/NayukiChiba/CNN/edit/main/docs/:path',
      text: '在 GitHub 上编辑此页',
    },

    lastUpdated: {
      text: '最后更新于',
    },
  },
})

import './style.css'
import DefaultTheme from 'vitepress/theme'
import { watch, nextTick, onMounted } from 'vue'
import { useRoute } from 'vitepress'
import mermaid from 'mermaid'

mermaid.initialize({
  startOnLoad: false,
  securityLevel: 'loose',
})

export default {
  extends: DefaultTheme,
  setup() {
    const route = useRoute()

    const renderMermaid = async () => {
      await nextTick()

      // 将隐藏的 <pre class="mermaid"> 替换为 <div class="mermaid">
      const preEls = document.querySelectorAll('pre.mermaid')
      preEls.forEach((el) => {
        const div = document.createElement('div')
        div.className = 'mermaid'
        div.textContent = el.textContent || ''
        el.replaceWith(div)
      })

      const mermaidEls = document.querySelectorAll('div.mermaid')
      if (mermaidEls.length === 0) return

      mermaidEls.forEach((el, i) => {
        if (!el.getAttribute('id')) {
          el.setAttribute('id', `mermaid-${Date.now()}-${i}`)
        }
      })

      try {
        await mermaid.run({ nodes: Array.from(mermaidEls) })
      } catch (e) {
        console.warn('Mermaid render error:', e)
      }
    }

    onMounted(renderMermaid)
    watch(() => route.path, renderMermaid)
  },
}

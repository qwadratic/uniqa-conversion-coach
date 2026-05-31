import { useEffect, useState } from 'react'

const LINKS = [
  { hash: '', label: 'Simulation Replay' },
  { hash: 'sessions', label: 'Session Logs' },
  { hash: 'docs', label: 'System Docs' },
  { hash: 'deck', label: 'Pitch Deck' },
]

export function useHash(): string {
  const [hash, setHash] = useState(window.location.hash.replace('#', ''))
  useEffect(() => {
    const h = () => setHash(window.location.hash.replace('#', ''))
    window.addEventListener('hashchange', h)
    return () => window.removeEventListener('hashchange', h)
  }, [])
  return hash
}

export default function Nav({ current }: { current: string }) {
  return (
    <nav className="nav">
      {LINKS.map(({ hash, label }) => (
        <a key={hash} href={`#${hash}`} className={current === hash ? 'active' : ''}>{label}</a>
      ))}
      <div className="nav-spacer" />
      <a href="https://github.com/qwadratic/uniqa-conversion-coach" target="_blank" rel="noopener" className="nav-ext">
        GitHub ↗
      </a>
    </nav>
  )
}

import { useEffect, useState } from 'react'
import type { Session } from './types'

async function loadJsonl<T>(url: string): Promise<T[]> {
  const text = await (await fetch(url)).text()
  return text.trim().split('\n').filter(Boolean).map((l) => JSON.parse(l) as T)
}

export default function SessionLogs() {
  const [off, setOff] = useState<Session[]>([])
  const [on, setOn] = useState<Session[]>([])
  const [arm, setArm] = useState<'off' | 'on'>('on')
  const [idx, setIdx] = useState(0)
  const [expanded, setExpanded] = useState<number | null>(null)

  useEffect(() => {
    const B = import.meta.env.BASE_URL
    ;(async () => {
      setOff(await loadJsonl<Session>(`${B}sessions_coach_off.jsonl`))
      setOn(await loadJsonl<Session>(`${B}sessions_coach_on.jsonl`))
    })().catch(console.error)
  }, [])

  const sessions = arm === 'on' ? on : off
  const session = sessions[idx]

  if (!sessions.length) return <div className="wrap">Loading sessions…</div>

  return (
    <div className="wrap">
      <h2>Raw Session Logs</h2>
      <p className="hint">Full simulation data for each session — every event, persona thought, mental state, coach decision, and intervention assessment.</p>

      <div className="controls">
        <div className="seg">
          {(['off', 'on'] as const).map((k) => (
            <button key={k} className={arm === k ? 'on' : ''} onClick={() => { setArm(k); setIdx(0) }}>Coach {k.toUpperCase()}</button>
          ))}
        </div>
        <select value={idx} onChange={(e) => setIdx(Number(e.target.value))}>
          {sessions.map((s, i) => <option key={i} value={i}>#{i + 1} · {s.persona} → {s.outcome} ({s.n_steps} steps)</option>)}
        </select>
      </div>

      {session && (
        <div className="session-detail">
          <div className="session-meta">
            <span className={`badge ${session.outcome}`}>{session.outcome}</span>
            <span className="meta-item">persona: <b>{session.persona}</b></span>
            <span className="meta-item">steps: <b>{session.n_steps}</b></span>
            <span className="meta-item">coach interventions: <b>{session.coach_interventions}</b></span>
          </div>

          {session.session_instance && (
            <details className="raw-block">
              <summary>Session Instance (per-session latent parameters)</summary>
              <pre>{JSON.stringify(session.session_instance, null, 2)}</pre>
            </details>
          )}

          <h3>Step-by-step trace</h3>
          {session.steps.map((st, si) => (
            <div key={si} className={`step-log ${expanded === si ? 'expanded' : ''}`}>
              <div className="step-log-head" onClick={() => setExpanded(expanded === si ? null : si)}>
                <span className="step-code">{st.step.split('_')[0]}</span>
                <span className="step-name">{st.step}</span>
                {st.persona_output?.feeling && <span className={`chip ${st.persona_output.feeling}`}>{st.persona_output.feeling}</span>}
                {st.persona_output?.decision && <span className="meta-item">→ {st.persona_output.decision}</span>}
                {st.coach_decision?._acted && <span className="coach-badge">⚡ {st.coach_decision.command?.effector}</span>}
                <span className="expand-icon">{expanded === si ? '▾' : '▸'}</span>
              </div>

              {expanded === si && (
                <div className="step-log-body">
                  <div className="log-section">
                    <h5>Persona Events</h5>
                    <div className="event-list">
                      {st.persona_output?.events?.map((ev, ei) => (
                        <div key={ei} className="event-row">
                          <span className="ev-t">t={ev.t}s</span>
                          <span className="ev-type">{ev.type}</span>
                          {ev.target && <span className="ev-target">{ev.target}</span>}
                          {ev.value != null && <span className="ev-val">{String(ev.value)}</span>}
                          {ev.thought && <div className="ev-thought">💭 {ev.thought}</div>}
                        </div>
                      ))}
                    </div>
                  </div>

                  {st.persona_output?.state && (
                    <div className="log-section">
                      <h5>Mental State</h5>
                      <pre>{JSON.stringify(st.persona_output.state, null, 2)}</pre>
                    </div>
                  )}

                  {st.persona_output?.reason && (
                    <div className="log-section">
                      <h5>Decision Reason</h5>
                      <p className="thought">"{st.persona_output.reason}"</p>
                    </div>
                  )}

                  {st.persona_output?.intervention_assessment && (
                    <div className="log-section">
                      <h5>Intervention Assessment</h5>
                      <pre>{JSON.stringify(st.persona_output.intervention_assessment, null, 2)}</pre>
                    </div>
                  )}

                  <div className="log-section">
                    <h5>Coach Decision</h5>
                    <pre>{JSON.stringify(st.coach_decision, null, 2)}</pre>
                  </div>
                </div>
              )}
            </div>
          ))}

          <details className="raw-block" style={{ marginTop: 16 }}>
            <summary>Full session JSON (raw)</summary>
            <pre className="raw-json">{JSON.stringify(session, null, 2)}</pre>
          </details>
        </div>
      )}
    </div>
  )
}

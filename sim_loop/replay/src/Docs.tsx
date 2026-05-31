export default function Docs() {
  return (
    <div className="wrap docs">
      <h2>System Documentation</h2>
      <p className="hint">How the simulation works — persona model, coach mechanics, hypotheses, interventions, and event types.</p>

      <div className="doc-toc">
        <a href="#docs-persona">Persona Model</a>
        <a href="#docs-coach">Coach Mechanics</a>
        <a href="#docs-hypotheses">Hypotheses</a>
        <a href="#docs-interventions">Interventions</a>
        <a href="#docs-events">Events & Signals</a>
      </div>

      {/* ─── PERSONA MODEL ─── */}
      <section id="docs-persona">
        <h3>Persona Model</h3>
        <p>Three persona segments, each with different conversion targets and behavioural profiles. Each session samples a <b>session instance</b> (latent per-session parameters) that makes each simulated user unique.</p>

        <h4>Segments</h4>
        <table className="doc-table">
          <thead><tr><th>Persona</th><th>Mix</th><th>Conversion Target</th><th>Key traits</th></tr></thead>
          <tbody>
            <tr><td><b>Judith</b> — Rising Hybrid</td><td>30%</td><td>Online purchase OR smooth advisor handoff (both count)</td><td>Deliberate researcher, high comprehension, advisor-affine, high commitment anxiety</td></tr>
            <tr><td><b>Franz</b> — Online Affine</td><td>50%</td><td>Online purchase ONLY (advisor = failure)</td><td>Fast, decisive, value-conscious, hates friction, drops at final price if it jumps</td></tr>
            <tr><td><b>Peter</b> — Service Affine</td><td>20%</td><td>Qualified service contact (callback/phone/WhatsApp)</td><td>Easily overwhelmed, low patience, mobile, wants a human before committing</td></tr>
          </tbody>
        </table>

        <h4>Behavioural Dials (static traits, per persona)</h4>
        <p>10 fundamental factors in [0,1], rendered to graded language (never raw numbers in the prompt). These are fixed dispositions that don't change within a session:</p>
        <table className="doc-table compact">
          <thead><tr><th>Dial</th><th>Drives</th><th>Judith</th><th>Franz</th><th>Peter</th></tr></thead>
          <tbody>
            <tr><td><code>budget_pressure</code></td><td>Price sensitivity / strain</td><td>0.14</td><td>0.56</td><td>0.68</td></tr>
            <tr><td><code>value_orientation</code></td><td>Price-performance scrutiny</td><td>0.68</td><td>0.89</td><td>0.50</td></tr>
            <tr><td><code>complexity_overwhelm</code></td><td>Give-up on complex screens</td><td>0.30</td><td>0.15</td><td>0.86</td></tr>
            <tr><td><code>advisor_lean</code></td><td>Preference for human contact</td><td>0.34</td><td>0.08</td><td>0.59</td></tr>
            <tr><td><code>patience</code></td><td>Tolerance for long forms</td><td>0.55</td><td>0.45</td><td>0.25</td></tr>
            <tr><td><code>ux_willingness</code></td><td>Push through heavy UI</td><td>0.55</td><td>0.55</td><td>0.27</td></tr>
            <tr><td><code>comprehension</code></td><td>Understanding under load</td><td>0.80</td><td>0.85</td><td>0.35</td></tr>
            <tr><td><code>distractibility</code></td><td>Susceptibility to external interruption</td><td>0.45</td><td>0.35</td><td>0.65</td></tr>
            <tr><td><code>commitment_anxiety</code></td><td>Hesitance to finalize</td><td>0.89</td><td>0.51</td><td>0.53</td></tr>
            <tr><td><code>uncertainty_aversion</code></td><td>Discomfort with unknowns</td><td>0.74</td><td>0.66</td><td>0.59</td></tr>
          </tbody>
        </table>

        <h4>Dynamic Mental State (evolves per step)</h4>
        <p>Tracked continuously and updated after perceiving each screen:</p>
        <ul>
          <li><code>attention</code> (0–1) — focus level; decays from distractions</li>
          <li><code>satisfaction</code> (0–1) — how well the experience meets expectations</li>
          <li><code>effort_left</code> (0–1) — remaining willingness to continue</li>
          <li><code>grasp</code> (0–1) — how much they understood the current screen</li>
          <li><code>effort_vs_reward</code> (0–1) — perceived payoff of continuing</li>
        </ul>

        <h4>Per-Session Instance (sampled, persona-weighted)</h4>
        <p>Each session rolls unique latent parameters that override the segment mean:</p>
        <ul>
          <li><code>time_pressure</code> — rushed / calm / curious</li>
          <li><code>visit_goal</code> — price-check / research / comparing / serious / ready-to-buy</li>
          <li><code>familiarity</code> — first time / been here before</li>
          <li><code>price_expectation</code> — cheap / flexible / expects premium</li>
          <li><code>advisor_need_today</code> — prefers online / wants reassurance / definitely wants a person</li>
          <li><code>screening_confidence</code> — comfortable with forms / wary of health questions</li>
          <li><code>coverage_need</code> — routine visits / dental / therapy / mental-health / specific condition / etc.</li>
          <li><code>open_question</code> — waiting periods / premium increases / network coverage / etc.</li>
          <li><code>recalls_measurements</code> — knows height/weight or not</li>
          <li><code>age</code> — determines real tariff price</li>
        </ul>

        <h4>Feelings / Exit Taxonomy</h4>
        <p>The persona decides to leave via one of these felt states:</p>
        <table className="doc-table compact">
          <thead><tr><th>Feeling</th><th>Layer</th><th>Trigger</th></tr></thead>
          <tbody>
            <tr><td><code>engaged</code></td><td>—</td><td>Screen meets expectations → continue</td></tr>
            <tr><td><code>dissatisfied</code></td><td>Conscious</td><td>Price &gt; expected, advisory wall, contradicting info</td></tr>
            <tr><td><code>cant_grasp</code></td><td>Subconscious</td><td>Low comprehension × high UX complexity → drift-off</td></tr>
            <tr><td><code>too_much_effort</code></td><td>Subconscious</td><td>Low willingness × high complexity → refuses without articulating why</td></tr>
            <tr><td><code>distracted</code></td><td>Exogenous</td><td>Life interruption pulls them away; may not return</td></tr>
            <tr><td><code>goal_achieved</code></td><td>Intent</td><td>Came to check the price; saw it; leaves content</td></tr>
            <tr><td><code>coverage_mismatch</code></td><td>Conscious</td><td>Their need is excluded / only in pricier tier</td></tr>
            <tr><td><code>unanswered_question</code></td><td>Conscious</td><td>Concrete question with no way to ask it on the interface</td></tr>
          </tbody>
        </table>

        <h4>Cognitive Model (per-step decision rules)</h4>
        <p>The persona prompt includes these emergent rules (no hard-coded churn rates):</p>
        <ul>
          <li><b>Price reaction:</b> Compare real tariff price (age-based) vs <code>price_expectation</code>, weighted by <code>budget_pressure</code> and <code>value_orientation × grasp</code>. Value-minded users do feasibility math (annual cost vs coverage limit vs realistic usage).</li>
          <li><b>Commitment rule:</b> At the final form (S6/S7): <code>commitment_anxiety</code> + <code>uncertainty_aversion</code> + effort drain + advisor lean. If the final price jumped (health loading), trigger the same price reaction with a delay (confusion first, then decide).</li>
          <li><b>Coverage reaction:</b> Check if <code>coverage_need</code> is met by available tariffs. If excluded → coverage_mismatch. If can't tell what differs → cant_grasp.</li>
          <li><b>Decision rule:</b> Leave when a state variable crosses tolerance, via the feeling that fired. <code>visit_goal</code> is the drive to push through — a price-checker leaves content once they see the number.</li>
        </ul>
      </section>

      {/* ─── COACH MECHANICS ─── */}
      <section id="docs-coach">
        <h3>Coach Mechanics</h3>
        <p>The coach is a <b>detection + decision</b> layer that sits on top of the immutable funnel. It watches observable events (never thoughts or mental state) and decides whether to show ONE widget per step.</p>

        <h4>5-Step Decision Workflow (every turn)</h4>
        <ol>
          <li><b>Persona Belief</b> — maintain confidence distribution over {'{'}judith, franz, peter{'}'}. Evidence: navigation speed, hesitation patterns, device, traffic source. Commit by S3/S4.</li>
          <li><b>Pains & Frustration</b> — infer from micro-signals: price shock (dwell after price reveal + cancel hover), form overwhelm (back-nav, re-edits), term confusion (text-select, tooltip opens), compare intent (tab-away + return).</li>
          <li><b>Dropout Likelihood + Temperature</b> — P(bounce this step). Willingness to act = temperature rising with dropout × belief confidence. Low → WAIT. High → act. Exit-intent spikes it.</li>
          <li><b>Widget Match</b> — pick ONE effector from the catalog that addresses the detected pain AND serves this persona's conversion target. Prefer least-intrusive pattern; escalate with temperature.</li>
          <li><b>Feedback</b> — prior widget dismissed → back off. Engaged → may follow up once.</li>
        </ol>

        <h4>Information Isolation</h4>
        <p>The coach NEVER sees: persona label, thoughts, mental-state variables, feeling, or health data. It only observes the event log (types, targets, timing, continue/leave decisions).</p>

        <h4>Budget & Constraints</h4>
        <ul>
          <li>≤3 widgets per session (annoyance budget)</li>
          <li>S1–S2 are detection-only (no interventions)</li>
          <li><code>form_simplify</code> ONLY on form steps (S3/S6), never on price screens</li>
          <li>One widget at a time; NO_ACTION is the default</li>
        </ul>

        <h4>Output Format</h4>
        <pre>{`{
  "persona_belief": {"judith": 0.2, "franz": 0.7, "peter": 0.1},
  "detected_pains": ["price_shock", "form_overwhelm"],
  "frustration": 0.4,
  "dropout_likelihood": 0.6,
  "intervention_temperature": 0.5,
  "reasoning": "signal → persona → why this widget now",
  "command": {
    "effector": "price_reframe",
    "category": "price",
    "fe_pattern": "price_chip",
    "surface": "on_page",
    "title": "€68/mo = €2.27/day",
    "message": "Less than a coffee...",
    "cta": "Choose Optimal",
    "target": null
  },
  "hypotheses": ["User is price-shocked"],
  "value_estimate": 0.7
}`}</pre>
      </section>

      {/* ─── HYPOTHESES ─── */}
      <section id="docs-hypotheses">
        <h3>Hypotheses Registry</h3>
        <p>Priors the coach reasons with — falsifiable beliefs, not rigid rules. Each is tested against the event trace.</p>
        <table className="doc-table">
          <thead><tr><th>ID</th><th>Hypothesis</th><th>Counter-move</th></tr></thead>
          <tbody>
            <tr><td>H1</td><td>Fast early steps → Franz (don't interrupt early; save budget for S6)</td><td>Wait / jump_to_pricing</td></tr>
            <tr><td>H2</td><td>Slow + overwhelm + back-nav → Peter (early warm handoff)</td><td>callback_offer / whatsapp_bot / contact_handoff</td></tr>
            <tr><td>H3</td><td>Deliberate research + hover/tooltip/compare → Judith (term help + comparison)</td><td>coverage_explain / faq_cards / advisor_handoff</td></tr>
            <tr><td>H4</td><td>Traffic source is a persona prior (paid search → Franz/Judith; display/social → Peter)</td><td>—</td></tr>
            <tr><td>H5</td><td>Fast-fill → offer jump_to_pricing (reward momentum)</td><td>jump_to_pricing</td></tr>
            <tr><td>H6</td><td>Confident tariff pick → high P(convert), don't over-intervene</td><td>NO_ACTION</td></tr>
            <tr><td>H7</td><td>Momentum is signal (steady = leave alone; stall/regress = act)</td><td>—</td></tr>
            <tr><td>H8</td><td>Price-table shock (freeze/cancel_hover/exit_intent) → persona-routed reframe</td><td>price_reframe / pricing_explain / value_justification</td></tr>
            <tr><td>H9</td><td>Leave-to-compare (tab_away→fast compare_return) → comparison table; long away → re-orient</td><td>market_comparison / save_progress</td></tr>
            <tr><td>H10</td><td>Term confusion (text_select/copy/scroll_up) → explain</td><td>coverage_explain / faq_cards</td></tr>
            <tr><td>H11</td><td>Forgot a field (stalls on SV number / weight) → defer</td><td>field_defer / form_helper</td></tr>
            <tr><td>H12</td><td>Big-form scare → pre-emptive explainer</td><td>form_explainer / form_simplify / bucket_input</td></tr>
            <tr><td>H13</td><td>Premium dead end (premium_click→nav_back) → "Optimal is fully online"</td><td>upgrade_explain</td></tr>
            <tr><td>H14</td><td>Pre-indicate price-affecting fields (price_preview) so price is never a surprise</td><td>price_preview</td></tr>
            <tr><td>H15</td><td>Bucket not exact → replace exact field with categories</td><td>bucket_input</td></tr>
            <tr><td>H16</td><td>Peter wants a human → early callback/whatsapp/contact_handoff</td><td>callback_offer / whatsapp_bot / contact_handoff</td></tr>
            <tr><td>H17</td><td>Mobile → phone capture (bottom sheet)</td><td>phone_capture</td></tr>
            <tr><td>H18</td><td>ID Austria login autofill removes biggest form friction</td><td>id_austria_login</td></tr>
            <tr><td>H19</td><td>Persist partial forms + save_progress / resume</td><td>save_progress</td></tr>
            <tr><td>H20</td><td>S5 add-on → suggest skipping (protects conversion from 24% drop)</td><td>addon_skip_ok</td></tr>
            <tr><td>H21</td><td>Tooltip hover = it helped → surface Like button; close = negative feedback</td><td>—</td></tr>
          </tbody>
        </table>
      </section>

      {/* ─── INTERVENTIONS ─── */}
      <section id="docs-interventions">
        <h3>Intervention Catalog (Full Effector Library)</h3>
        <p>32 typed interventions the coach can deploy. Each has a category, default frontend pattern, and applicable steps/personas.</p>

        <h4>Categories</h4>
        <div className="cat-grid">
          <span className="cat-pill" style={{ background: '#0046A0' }}>€ price</span>
          <span className="cat-pill" style={{ background: '#2563EB' }}>i inform</span>
          <span className="cat-pill" style={{ background: '#1FA971' }}>✓ reassure</span>
          <span className="cat-pill" style={{ background: '#7C3AED' }}>✦ engage</span>
          <span className="cat-pill" style={{ background: '#0891B2' }}>→ convert_aid</span>
          <span className="cat-pill" style={{ background: '#F0A028' }}>✉ capture</span>
          <span className="cat-pill" style={{ background: '#E2001A' }}>☎ handoff</span>
        </div>

        <h4>Frontend Patterns</h4>
        <p><code>price_chip</code> · <code>card</code> · <code>popover</code> · <code>banner</code> · <code>inline_expand</code> · <code>toast</code> · <code>bottom_sheet</code> · <code>progress_ribbon</code> · <code>sticky_bar</code> · <code>chip</code></p>

        <h4>Full List</h4>
        <table className="doc-table compact">
          <thead><tr><th>ID</th><th>Category</th><th>Pattern</th><th>Steps</th><th>Fits</th><th>What the user sees</th></tr></thead>
          <tbody>
            {interventionData.map((e) => (
              <tr key={e.id}>
                <td><code>{e.id}</code></td>
                <td>{e.cat}</td>
                <td>{e.fe}</td>
                <td>{e.apt}</td>
                <td>{e.who || 'all'}</td>
                <td>{e.body}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      {/* ─── EVENTS & SIGNALS ─── */}
      <section id="docs-events">
        <h3>Events & Signals</h3>
        <p>All event types the simulation can generate, and the derived signals the coach observes.</p>

        <h4>Atomic Event Types</h4>
        <table className="doc-table compact">
          <thead><tr><th>Event</th><th>Description</th></tr></thead>
          <tbody>
            {eventTypes.map(([ev, desc]) => (
              <tr key={ev}><td><code>{ev}</code></td><td>{desc}</td></tr>
            ))}
          </tbody>
        </table>

        <h4>Derived Signals (scored over time, not single events)</h4>
        <table className="doc-table compact">
          <thead><tr><th>Category</th><th>Signals</th></tr></thead>
          <tbody>
            <tr><td><b>Hesitation / Friction</b></td><td>back_nav_count, field_reedit_count, validation_error_count, field_clear_count, longest_idle_sec, time_to_first_action, hesitation (0..1)</td></tr>
            <tr><td><b>Info-seeking / Confusion</b></td><td>tooltip_open_count, hover_count, scroll_reversals, text_select_count, term_copy_count, slow_mouse_ratio</td></tr>
            <tr><td><b>Comparison / Leaving</b></td><td>tab_away_count, tab_away_total_sec, return_time, external_nav_count, exit_intent_count</td></tr>
            <tr><td><b>Price Reaction</b></td><td>price_hover_count, cancel_hover_count, dwell-after-price_reveal</td></tr>
            <tr><td><b>Effort / Momentum</b></td><td>keystroke_total, taps_total, step_dwell_sec, step_revisit_count, momentum, total_session_sec</td></tr>
          </tbody>
        </table>

        <h4>Terminal Events</h4>
        <table className="doc-table compact">
          <thead><tr><th>Event</th><th>Meaning</th><th>Win for</th></tr></thead>
          <tbody>
            <tr><td><code>convert</code></td><td>Online purchase completed</td><td>Franz (required), Judith (counts), Peter (bonus)</td></tr>
            <tr><td><code>advisor_booked</code></td><td>Smooth advisor handoff</td><td>Judith</td></tr>
            <tr><td><code>callback_booked</code></td><td>Phone callback scheduled</td><td>Peter</td></tr>
            <tr><td><code>contact_clicked</code></td><td>Phone/WhatsApp/service contact engaged</td><td>Peter</td></tr>
            <tr><td><code>abandon</code></td><td>Left without converting</td><td>Nobody (loss for all)</td></tr>
          </tbody>
        </table>
      </section>
    </div>
  )
}

// ─── Data ────────────────────────────────────────────────────────────────────

const interventionData = [
  { id: 'price_reframe', cat: 'price', fe: 'price_chip', apt: 'S4,S7', who: '', body: '€/day reframe — "less than a coffee, for what it covers"' },
  { id: 'pricing_explain', cat: 'price', fe: 'card', apt: 'S4,S7', who: '', body: 'How your premium is built: age + tariff. Health may adjust final price — nothing hidden.' },
  { id: 'price_preview', cat: 'price', fe: 'popover', apt: 'S6', who: '', body: 'Before you fill: here\'s the small impact this answer has on your price.' },
  { id: 'value_justification', cat: 'price', fe: 'card', apt: 'S7', who: 'franz,judith', body: 'Final price higher? Here\'s why + a comparable cheaper tariff, still fully online.' },
  { id: 'health_explain', cat: 'inform', fe: 'banner', apt: 'S7', who: '', body: 'Final price moved after health questions — here\'s why, still fully online.' },
  { id: 'upgrade_explain', cat: 'inform', fe: 'banner', apt: 'S4', who: 'franz,judith', body: 'Premium needs advisor — but Optimal is fully completable online right now.' },
  { id: 'package_nuance', cat: 'inform', fe: 'inline_expand', apt: 'S4', who: '', body: 'Start vs Optimal — 3 real differences (limits, refund %, inclusions). No upsell.' },
  { id: 'coverage_explain', cat: 'inform', fe: 'popover', apt: 'S1,S4', who: '', body: 'What limits mean in real life: ≈ X specialist visits or a physio series/year.' },
  { id: 'coverage_checker', cat: 'inform', fe: 'popover', apt: 'S1,S4', who: '', body: 'Is YOUR treatment/doctor covered? The one question the form can\'t answer.' },
  { id: 'faq_cards', cat: 'inform', fe: 'card', apt: 'S1,S4,S5', who: '', body: 'Quick answers to what you\'re wondering — no need to leave and search.' },
  { id: 'feature_highlight', cat: 'inform', fe: 'toast', apt: 'S4', who: '', body: 'Recently improved: e.g. laser-eye limit doubled in 2025.' },
  { id: 'trust_signal', cat: 'reassure', fe: 'banner', apt: 'any', who: '', body: 'UNIQA — insuring Austria since 1811, AAA-rated.' },
  { id: 'social_proof', cat: 'reassure', fe: 'banner', apt: 'S4,S7', who: 'judith,peter', body: 'Most people with your needs chose Optimal this month.' },
  { id: 'addon_skip_ok', cat: 'reassure', fe: 'banner', apt: 'S5', who: '', body: 'Add-ons are optional — skipping keeps your price, add any later.' },
  { id: 'quick_quiz', cat: 'engage', fe: 'bottom_sheet', apt: 'S1,S4', who: 'peter,judith', body: 'Not sure which tariff? 3 quick questions → recommendation (~60s).' },
  { id: 'form_simplify', cat: 'engage', fe: 'bottom_sheet', apt: 'S3,S6', who: 'peter', body: 'Only required fields, split into small steps.' },
  { id: 'field_defer', cat: 'engage', fe: 'popover', apt: 'S3,S6', who: '', body: 'Don\'t have it? Skip now, add later — we flag if it affects price.' },
  { id: 'bucket_input', cat: 'engage', fe: 'popover', apt: 'S6', who: '', body: 'Pick a range instead of exact — price impact per range shown.' },
  { id: 'form_helper', cat: 'engage', fe: 'popover', apt: 'S3,S6', who: '', body: 'Your SV number is top-right on your e-card.' },
  { id: 'form_explainer', cat: 'engage', fe: 'progress_ribbon', apt: 'S3,S6', who: '', body: '~4 fields (~1 min) → binding price. Then done.' },
  { id: 'jump_to_pricing', cat: 'convert_aid', fe: 'sticky_bar', apt: 'S2,S3', who: 'franz', body: 'Moving fast — skip ahead to your price now?' },
  { id: 'id_austria_login', cat: 'engage', fe: 'card', apt: 'S3,S6', who: '', body: 'Auto-fill with ID Austria — no typing SV number.' },
  { id: 'preselect_optimal', cat: 'convert_aid', fe: 'chip', apt: 'S4', who: 'franz', body: 'Optimal pre-selected as sensible default — change anytime.' },
  { id: 'upgrade_path', cat: 'convert_aid', fe: 'banner', apt: 'S4', who: 'judith', body: 'Start lower, upgrade within 3 years — no new health check.' },
  { id: 'save_progress', cat: 'capture', fe: 'sticky_bar', apt: 'S4,S7', who: '', body: 'Not finishing today? Save to email link, pick up here later.' },
  { id: 'email_capture', cat: 'capture', fe: 'card', apt: 'S4,S7', who: '', body: 'Want this quote in writing? We email summary + explainer.' },
  { id: 'phone_capture', cat: 'capture', fe: 'bottom_sheet', apt: 'S4,S7', who: '', body: 'On your phone? Get quote by text + callback.' },
  { id: 'callback_offer', cat: 'handoff', fe: 'bottom_sheet', apt: 'S1,S3', who: 'peter', body: 'Prefer a person? Book a free callback.' },
  { id: 'whatsapp_bot', cat: 'handoff', fe: 'bottom_sheet', apt: 'S1,S3,S4', who: 'peter', body: 'Ask on WhatsApp — answers + quote sent there.' },
  { id: 'contact_handoff', cat: 'handoff', fe: 'bottom_sheet', apt: 'S3,S4,S6', who: 'peter', body: 'Don\'t fill any of this — leave email/phone, we take it from here.' },
  { id: 'voice_questions', cat: 'handoff', fe: 'bottom_sheet', apt: 'S1,S3,S4', who: 'peter', body: 'Leave a number for callback, or type your questions.' },
  { id: 'advisor_handoff', cat: 'handoff', fe: 'bottom_sheet', apt: 'S7', who: 'judith', body: 'Talk to an advisor before committing — fully optional.' },
]

const eventTypes: [string, string][] = [
  ['step_enter', 'User entered a new funnel step'],
  ['mouse_move', 'Low-level cursor movement (collapsed by event processor)'],
  ['hover', 'Cursor dwell over an element'],
  ['pause', 'No input for a while (micro-pause)'],
  ['idle', 'Extended dwell with no input'],
  ['scroll', 'Page scroll'],
  ['scroll_up', 'Scrolled back up to re-read (confusion signal)'],
  ['field_focus', 'Focused on a form field'],
  ['field_blur', 'Left a form field'],
  ['field_edit', 'Changed a field value'],
  ['field_invalid', 'Submitted invalid field value'],
  ['field_clear', 'Cleared a field after typing (doubt)'],
  ['keystroke', 'Keystrokes typed into a field (UX cost)'],
  ['tap', 'Taps/clicks on a target'],
  ['select', 'Chose an option (dropdown/choice card)'],
  ['select_card', 'Selected a choice card'],
  ['dropdown_open', 'Opened a dropdown'],
  ['open_dropdown', 'Opened a dropdown (variant)'],
  ['filter_type', 'Typed in a dropdown filter'],
  ['select_option', 'Selected a dropdown option'],
  ['tooltip_open', 'Opened an info tooltip'],
  ['validation_error', 'Inline field error shown'],
  ['price_reveal', 'A tariff price became visible'],
  ['price_hover', 'Hovered over a price element'],
  ['tariff_click', 'Clicked a tariff option'],
  ['premium_click', 'Clicked an advisory-only tariff'],
  ['nav_back', 'Navigated backwards'],
  ['nav_next', 'Clicked next/continue'],
  ['session_gap', 'Returned after a long pause'],
  ['tab_blur', 'Switched to another tab (attention left)'],
  ['tab_focus', 'Re-activated this tab'],
  ['cancel_hover', 'Hovered over cancel/close element'],
  ['exit_intent', 'Cursor darted toward close/URL bar (leaving NOW)'],
  ['text_select', 'Highlighted text (reading hard / selecting jargon)'],
  ['copy', 'Copied text (about to google / leave to compare)'],
  ['external_nav', 'Opened a new tab / left to compare'],
  ['compare_return', 'Returned from comparison; value = seconds away'],
  ['slow_mouse', 'Slow wandering cursor (deliberation/uncertainty)'],
  ['rage_click', 'Repeated clicks same target (frustration)'],
  ['widget_shown', 'Coach widget displayed'],
  ['widget_cta', 'User clicked the widget CTA (engaged)'],
  ['widget_dismiss', 'User closed the widget (back off)'],
  ['widget_like', 'Thumbs-up on widget (positive feedback)'],
  ['widget_dislike', 'Thumbs-down (change tactic)'],
  ['submit', 'Form submitted'],
  ['abandon', 'Left without converting (terminal)'],
  ['convert', 'Online purchase completed (terminal)'],
  ['advisor_booked', 'Smooth advisor handoff booked (terminal for Judith)'],
  ['callback_booked', 'Phone callback scheduled (terminal for Peter)'],
  ['contact_clicked', 'Service contact engaged (terminal for Peter)'],
]

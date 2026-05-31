export default function Deck() {
  const base = import.meta.env.BASE_URL
  return (
    <div className="wrap">
      <h2>Pitch Deck</h2>
      <p className="hint">The UNIQA Conversion Coach presentation deck — embedded below or <a href={`${base}uniqa_conversion_coach_deck.pdf`} target="_blank" rel="noopener">open in a new tab ↗</a></p>
      <div className="pdf-embed">
        <object
          data={`${base}uniqa_conversion_coach_deck.pdf`}
          type="application/pdf"
          width="100%"
          height="100%"
        >
          <p>Your browser doesn't support embedded PDFs. <a href={`${base}uniqa_conversion_coach_deck.pdf`} target="_blank" rel="noopener">Download the deck ↗</a></p>
        </object>
      </div>
    </div>
  )
}

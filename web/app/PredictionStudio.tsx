"use client";

import { useEffect, useMemo, useState } from "react";
import { professionalPlayers } from "../data/players.generated";
import { customProfile, predictMatch, type PlayerProfile, type PredictionResult, type Surface, type Tour } from "../lib/model";
import { defaultRatings, factorGuidance, interpretation, type CustomRatings, type FactorKey } from "../lib/profile-guidance";

type Mode = "tour" | "custom";
type CustomSide = "one" | "two";

const proPlayers: PlayerProfile[] = professionalPlayers.map((player) => ({
  ...player,
  surfaceRating: { ...player.surfaceRating },
  surfaceSamples: { ...player.surfaceSamples },
}));

const surfaceLabels: Record<Surface, string> = { hard: "Hard", clay: "Clay", grass: "Grass" };
const percent = (value: number, digits = 0) => `${(value * 100).toFixed(digits)}%`;

export function PredictionStudio() {
  const [mode, setMode] = useState<Mode>("tour");
  const [tour, setTour] = useState<Tour>("ATP");
  const [firstId, setFirstId] = useState("atp-1");
  const [secondId, setSecondId] = useState("atp-2");
  const [surface, setSurface] = useState<Surface>("hard");
  const [bestOf, setBestOf] = useState<3 | 5>(3);
  const [customNames, setCustomNames] = useState({ one: "Player A", two: "Player B" });
  const [customRatings, setCustomRatings] = useState<{ one: CustomRatings; two: CustomRatings }>({
    one: { ...defaultRatings, serve: 8, form: 7 },
    two: { ...defaultRatings, return: 8, movement: 8 },
  });
  const [activeCustom, setActiveCustom] = useState<CustomSide>("one");
  const [result, setResult] = useState<PredictionResult | null>(null);
  const [running, setRunning] = useState(false);

  const pool = useMemo(() => proPlayers.filter((player) => player.tour === tour), [tour]);
  const proOne = proPlayers.find((player) => player.id === firstId) ?? pool[0];
  const proTwo = proPlayers.find((player) => player.id === secondId) ?? pool[1];
  const customOne = useMemo(() => customProfile("custom-one", customNames.one, customRatings.one, surface), [customNames.one, customRatings.one, surface]);
  const customTwo = useMemo(() => customProfile("custom-two", customNames.two, customRatings.two, surface), [customNames.two, customRatings.two, surface]);
  const playerOne = mode === "tour" ? proOne : customOne;
  const playerTwo = mode === "tour" ? proTwo : customTwo;

  useEffect(() => {
    const timer = window.setTimeout(() => setResult(predictMatch(proPlayers[0], proPlayers[1], "hard", 3)), 120);
    return () => window.clearTimeout(timer);
  }, []);

  function invalidate() { setResult(null); }

  function changeMode(next: Mode) {
    setMode(next);
    setBestOf(3);
    setResult(null);
  }

  function changeTour(next: Tour) {
    const nextPool = proPlayers.filter((player) => player.tour === next);
    setTour(next);
    setFirstId(nextPool[0].id);
    setSecondId(nextPool[1].id);
    setResult(null);
  }

  function runForecast() {
    setRunning(true);
    window.setTimeout(() => {
      setResult(predictMatch(playerOne, playerTwo, surface, bestOf));
      setRunning(false);
      window.setTimeout(() => document.getElementById("forecast")?.scrollIntoView({ behavior: "smooth", block: "start" }), 30);
    }, 70);
  }

  function updateCustom(side: CustomSide, key: FactorKey, value: number) {
    setCustomRatings((current) => ({ ...current, [side]: { ...current[side], [key]: value } }));
    setResult(null);
  }

  return (
    <main>
      <nav className="topbar" aria-label="Primary navigation">
        <a className="brand" href="#top" aria-label="Baseline home"><span className="brand-mark" aria-hidden="true"><i /><i /><i /></span><span>BASELINE</span><em>LABS</em></a>
        <div className="nav-links"><a href="#studio">Match studio</a><a href="#method">Methodology</a><a href="#data">Data</a></div>
        <span className="model-badge"><i /> Model v2.0</span>
      </nav>

      <section className="hero" id="top">
        <div className="eyebrow"><span>Decision intelligence</span><i /> Tennis, modeled point by point</div>
        <h1>See the match<br />before it unfolds.</h1>
        <p className="hero-copy">A probabilistic tennis lab that treats ability as uncertain, adapts it to the court, then plays the match thousands of times—point by point.</p>
        <a className="hero-cta" href="#studio">Open match studio <span>↓</span></a>
        <div className="hero-proof"><div><strong>5,040</strong><span>score-level simulations</span></div><div><strong>36</strong><span>posterior skill draws</span></div><div><strong>32</strong><span>current tour profiles</span></div></div>
        <div className="court-orbit" aria-hidden="true"><span className="ball" /><span className="orbit-label">MONTE CARLO</span></div>
      </section>

      <section className="studio-shell" id="studio">
        <header className="studio-header"><div><span className="section-number">01 / MATCH STUDIO</span><h2>Build a matchup</h2></div><p>Start with ranked professionals or describe two players from first principles. Both routes feed the same scoring simulator.</p></header>
        <div className="mode-switch" role="tablist" aria-label="Prediction mode">
          <button className={mode === "tour" ? "active" : ""} onClick={() => changeMode("tour")} role="tab" aria-selected={mode === "tour"}><span>01</span> Professional players <small>Current ATP & WTA profiles</small></button>
          <button className={mode === "custom" ? "active" : ""} onClick={() => changeMode("custom")} role="tab" aria-selected={mode === "custom"}><span>02</span> Build your own <small>Guided 1–10 scouting inputs</small></button>
        </div>

        {mode === "tour" ? (
          <ProfessionalBuilder pool={pool} tour={tour} playerOne={proOne} playerTwo={proTwo} firstId={firstId} secondId={secondId} surface={surface} bestOf={bestOf} onTour={changeTour} onFirst={(id) => { setFirstId(id); invalidate(); }} onSecond={(id) => { setSecondId(id); invalidate(); }} onSurface={(value) => { setSurface(value); invalidate(); }} onBestOf={(value) => { setBestOf(value); invalidate(); }} onRun={runForecast} running={running} />
        ) : (
          <CustomBuilder active={activeCustom} names={customNames} ratings={customRatings} surface={surface} bestOf={bestOf} onActive={setActiveCustom} onName={(side, name) => { setCustomNames((current) => ({ ...current, [side]: name })); invalidate(); }} onRating={updateCustom} onSurface={(value) => { setSurface(value); invalidate(); }} onBestOf={(value) => { setBestOf(value); invalidate(); }} onRun={runForecast} running={running} />
        )}

        <ForecastPanel result={result} one={playerOne} two={playerTwo} surface={surface} />
      </section>

      <Methodology />
      <footer id="data">
        <a className="brand footer-brand" href="#top"><span className="brand-mark" aria-hidden="true"><i /><i /><i /></span><span>BASELINE</span><em>LABS</em></a>
        <p>Research software for probabilistic exploration. Not betting, medical, or financial advice. No affiliation with the ATP or WTA.</p>
        <div><a href="https://www.atptour.com/en/rankings/singles" target="_blank" rel="noreferrer">ATP rankings ↗</a><a href="https://www.wtatennis.com/rankings/singles" target="_blank" rel="noreferrer">WTA rankings ↗</a><a href="https://github.com/JeffSackmann" target="_blank" rel="noreferrer">Historical data attribution ↗</a></div>
      </footer>
    </main>
  );
}

function ProfessionalBuilder({ pool, tour, playerOne, playerTwo, firstId, secondId, surface, bestOf, onTour, onFirst, onSecond, onSurface, onBestOf, onRun, running }: {
  pool: PlayerProfile[]; tour: Tour; playerOne: PlayerProfile; playerTwo: PlayerProfile; firstId: string; secondId: string; surface: Surface; bestOf: 3 | 5;
  onTour: (tour: Tour) => void; onFirst: (id: string) => void; onSecond: (id: string) => void; onSurface: (surface: Surface) => void; onBestOf: (bestOf: 3 | 5) => void; onRun: () => void; running: boolean;
}) {
  return <div className="match-card">
    <div className="match-toolbar"><div className="segmented" aria-label="Tour">{(["ATP", "WTA"] as Tour[]).map((item) => <button key={item} className={tour === item ? "selected" : ""} onClick={() => onTour(item)}>{item}</button>)}</div><div className="data-freshness"><i /> Rankings verified · Aug 18, 2026</div></div>
    <div className="matchup-grid">
      <PlayerPicker label="Player A" player={playerOne} players={pool} value={firstId} blocked={secondId} surface={surface} onChange={onFirst} />
      <div className="versus"><span>VS</span><i /></div>
      <PlayerPicker label="Player B" player={playerTwo} players={pool} value={secondId} blocked={firstId} surface={surface} onChange={onSecond} align="right" />
    </div>
    <ConditionBar surface={surface} bestOf={bestOf} onSurface={onSurface} onBestOf={onBestOf} onRun={onRun} running={running} />
    <div className="data-disclosure"><strong>Data lens</strong><span>Official ranking prior as of Aug 18, 2026</span><span>Match history through May 25, 2026</span><span>Uncertainty grows where surface samples are thin</span></div>
  </div>;
}

function PlayerPicker({ label, player, players, value, blocked, surface, onChange, align = "left" }: {
  label: string; player: PlayerProfile; players: PlayerProfile[]; value: string; blocked: string; surface: Surface; onChange: (value: string) => void; align?: "left" | "right";
}) {
  const surfaceMatches = player.surfaceSamples[surface];
  return <article className={`player-picker ${align}`}>
    <div className="player-label">{label}</div><div className="rank-chip">WORLD NO. {player.rank}</div>
    <div className="player-monogram" aria-hidden="true">{player.name.split(" ").map((part) => part[0]).join("")}</div>
    <select value={value} onChange={(event) => onChange(event.target.value)} aria-label={`${label} player`}>{players.map((option) => <option key={option.id} value={option.id} disabled={option.id === blocked}>#{option.rank} · {option.name}</option>)}</select>
    <div className="player-meta"><span>{player.country} · {player.hand}-handed</span><span>{player.rankingPoints?.toLocaleString()} pts</span></div>
    <div className="profile-stats">
      <div><strong>{Math.round(player.surfaceRating[surface])}</strong><span>{surfaceLabels[surface]} rating</span></div>
      <div><strong>{percent(player.servePointsWon, 1)}</strong><span>Serve pts won</span></div>
      <div><strong>{player.wins52w}–{player.matches52w - player.wins52w}</strong><span>52-week record</span></div>
      <div><strong>{surfaceMatches}</strong><span>Surface sample</span></div>
    </div>
  </article>;
}

function CustomBuilder({ active, names, ratings, surface, bestOf, onActive, onName, onRating, onSurface, onBestOf, onRun, running }: {
  active: CustomSide; names: Record<CustomSide, string>; ratings: Record<CustomSide, CustomRatings>; surface: Surface; bestOf: 3 | 5;
  onActive: (side: CustomSide) => void; onName: (side: CustomSide, name: string) => void; onRating: (side: CustomSide, key: FactorKey, value: number) => void; onSurface: (surface: Surface) => void; onBestOf: (bestOf: 3 | 5) => void; onRun: () => void; running: boolean;
}) {
  const activeName = names[active].trim() || (active === "one" ? "Player A" : "Player B");
  return <div className="custom-card">
    <div className="custom-intro"><div><span className="section-number">GUIDED PROFILE BUILDER</span><h3>Translate what you see on court into model inputs.</h3></div><p>The numbers are not added together as weights. Each one changes a probability distribution, uncertainty term, or match-state behavior in the simulator.</p></div>
    <div className="custom-player-tabs">
      {(["one", "two"] as CustomSide[]).map((side, index) => <button key={side} className={active === side ? "active" : ""} onClick={() => onActive(side)}><span>PLAYER {index === 0 ? "A" : "B"}</span><strong>{names[side] || `Player ${index === 0 ? "A" : "B"}`}</strong><i>{Math.round(Object.values(ratings[side]).reduce((sum, value) => sum + value, 0) / 8 * 10) / 10} avg</i></button>)}
    </div>
    <div className="custom-name-row"><label htmlFor="custom-name">Player name</label><input id="custom-name" value={names[active]} onChange={(event) => onName(active, event.target.value)} placeholder={active === "one" ? "Player A" : "Player B"} /><p>Set every factor with the anchors below. The explanation updates to describe exactly what your selection implies.</p></div>
    <div className="factor-grid">
      {(Object.keys(factorGuidance) as FactorKey[]).map((key) => <FactorControl key={key} factorKey={key} score={ratings[active][key]} playerName={activeName} onChange={(value) => onRating(active, key, value)} />)}
    </div>
    <ConditionBar surface={surface} bestOf={bestOf} onSurface={onSurface} onBestOf={onBestOf} onRun={onRun} running={running} />
  </div>;
}

function FactorControl({ factorKey, score, playerName, onChange }: { factorKey: FactorKey; score: number; playerName: string; onChange: (value: number) => void }) {
  const guidance = factorGuidance[factorKey];
  return <article className="factor-card">
    <header><div><span>{guidance.label}</span><p>{guidance.short}</p></div><strong>{score}<small>/10</small></strong></header>
    <div className="slider-wrap"><input type="range" min="1" max="10" step="1" value={score} onChange={(event) => onChange(Number(event.target.value))} aria-label={`${guidance.label} for ${playerName}`} style={{ "--score": `${(score - 1) / 9 * 100}%` } as React.CSSProperties} /><div className="range-numbers">{Array.from({ length: 10 }, (_, index) => <span key={index}>{index + 1}</span>)}</div></div>
    <details><summary>See scoring examples <span>+</span></summary><div className="anchor-examples"><p>{guidance.low}</p><p>{guidance.middle}</p><p>{guidance.high}</p></div></details>
    <div className="selection-reading"><span>YOUR SELECTION</span><p>{interpretation(factorKey, score, playerName)}</p></div>
  </article>;
}

function ConditionBar({ surface, bestOf, onSurface, onBestOf, onRun, running }: { surface: Surface; bestOf: 3 | 5; onSurface: (surface: Surface) => void; onBestOf: (bestOf: 3 | 5) => void; onRun: () => void; running: boolean }) {
  return <div className="conditions"><span className="condition-label">Surface</span><div className="surface-options">{(["hard", "clay", "grass"] as Surface[]).map((item) => <button key={item} className={surface === item ? "selected" : ""} onClick={() => onSurface(item)}><i className={item} />{surfaceLabels[item]}</button>)}</div><span className="condition-label">Format</span><div className="format-options">{([3, 5] as const).map((item) => <button key={item} className={bestOf === item ? "selected" : ""} onClick={() => onBestOf(item)}>Best of {item}</button>)}</div><button className="run-button" onClick={onRun} disabled={running}>{running ? "Simulating match…" : "Run 5,040 simulations"}<span>{running ? "◌" : "↗"}</span></button></div>;
}

function ForecastPanel({ result, one, two, surface }: { result: PredictionResult | null; one: PlayerProfile; two: PlayerProfile; surface: Surface }) {
  if (!result) return <section className="forecast-empty" id="forecast"><div className="forecast-orb">?</div><div><span className="section-number">FORECAST WAITING</span><h3>Set the matchup, then run the model.</h3><p>The engine will resample player ability, simulate real tennis scoring, and report both the central forecast and its uncertainty.</p></div></section>;
  const winnerIsOne = result.projectedWinner === one.name;
  const winnerProbability = winnerIsOne ? result.playerOneProbability : result.playerTwoProbability;
  const low = winnerIsOne ? result.intervalLow : 1 - result.intervalHigh;
  const high = winnerIsOne ? result.intervalHigh : 1 - result.intervalLow;
  return <section className="forecast-panel" id="forecast">
    <header className="forecast-title"><div><span className="section-number">02 / MODEL OUTPUT</span><h2>The forecast</h2></div><div className={`confidence ${result.confidence.toLowerCase()}`}><i /> {result.confidence} evidence</div></header>
    <div className="winner-card">
      <div className="winner-copy"><span>PROJECTED WINNER · {surfaceLabels[surface].toUpperCase()}</span><h3>{result.projectedWinner}</h3><p>Wins in <strong>{percent(winnerProbability)}</strong> of score-level simulations, with an 80% posterior range of <strong>{percent(low)}–{percent(high)}</strong>.</p></div>
      <div className="win-prob"><span>WIN PROBABILITY</span><strong>{percent(winnerProbability)}</strong><small>not a certainty</small></div>
      <div className="probability-split"><i style={{ width: `${result.playerOneProbability * 100}%` }} /><span className="one-label">{one.name} · {percent(result.playerOneProbability)}</span><span className="two-label">{percent(result.playerTwoProbability)} · {two.name}</span></div>
    </div>
    <div className="forecast-kpis">
      <div><span>Likely set score</span><strong>{result.likelySetScore}</strong><small>winner’s perspective</small></div>
      <div><span>Expected match length</span><strong>{result.expectedSets.toFixed(2)}</strong><small>sets · {result.expectedGames.toFixed(1)} games</small></div>
      <div><span>At least one tiebreak</span><strong>{percent(result.tieBreakChance)}</strong><small>across simulated matches</small></div>
      <div><span>Serve-point outlook</span><strong>{percent(result.averageServePointOne, 1)} / {percent(result.averageServePointTwo, 1)}</strong><small>{one.name.split(" ").at(-1)} / {two.name.split(" ").at(-1)}</small></div>
    </div>
    <div className="evidence-panel"><div className="evidence-heading"><div><span className="section-number">EVIDENCE LAYERS</span><h3>Why the model leans this way</h3></div><p>These are diagnostics from separate model layers—not points in a weighted checklist.</p></div><div className="evidence-list">{result.evidence.map((item) => <div className="evidence-row" key={item.label}><div><strong>{item.label}</strong><span>{item.detail}</span></div><div className={`edge-track ${item.leader}`}><i style={{ width: `${Math.max(8, item.strength * 50)}%` }} /></div><b>{item.leader === "one" ? one.name.split(" ").at(-1) : item.leader === "two" ? two.name.split(" ").at(-1) : "Even"}</b></div>)}</div></div>
    <div className="forecast-foot"><span>{result.simulations.toLocaleString()} match simulations</span><span>{result.posteriorDraws} independent skill draws</span><span>80% model-uncertainty interval</span><span>Deterministic research seed</span></div>
  </section>;
}

function Methodology() {
  const layers = [
    ["01", "Dynamic ability", "A sequential Elo-style paired-comparison rating learns from wins, losses, and opponent strength. Official ranking points act as a current prior, not the answer."],
    ["02", "Surface pooling", "Hard, clay, and grass ratings learn separately, then shrink toward overall ability when the surface sample is small."],
    ["03", "Bayesian skills", "Serve and return point rates are resampled from beta posteriors. Rating ability is resampled from a normal posterior whose width reflects evidence volume."],
    ["04", "Tennis scoring", "Every simulated match plays points into games, advantage games, 6–6 tiebreaks, sets, and the selected best-of format."],
  ];
  return <section className="methodology" id="method">
    <header><div><span className="section-number">03 / METHODOLOGY</span><h2>Mechanics before mystique.</h2></div><p>A forecast should reveal where its certainty comes from—and where it does not.</p></header>
    <div className="method-grid">{layers.map(([number, title, copy]) => <article key={number}><span>{number}</span><h3>{title}</h3><p>{copy}</p></article>)}</div>
    <div className="research-note"><div><span>RESEARCH BASIS</span><h3>Built from methods used in serious paired-comparison and tennis forecasting work.</h3></div><p>Surface-aware dynamic paired comparison has beaten standard Elo and Glicko on held-out tennis log loss in published research. This implementation borrows that architecture—dynamic latent strength, covariates, partial pooling, and explicit uncertainty—without claiming identical calibration.</p><a href="https://arxiv.org/abs/1902.07378" target="_blank" rel="noreferrer">Read the paired-comparison paper ↗</a></div>
    <div className="limits"><strong>Responsible-use note</strong><p>This is a transparent research model, not a betting product. Rankings are newer than the historical match snapshot; injuries, live conditions, travel, coaching changes, and same-day news are not automatically ingested. Treat the interval as a model-uncertainty estimate—not a guarantee or a validated market edge.</p></div>
  </section>;
}

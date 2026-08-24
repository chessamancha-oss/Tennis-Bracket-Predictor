"use client";

import { useEffect, useMemo, useState } from "react";
import { playerDatabaseSummary } from "../data/player-database-summary.generated";
import { professionalPlayers } from "../data/players.generated";
import { forecastBracket, type BracketParticipant, type ForecastBracket } from "../lib/bracket";
import { advancedProfile, predictMatch, type AdvancedProfileInputs, type PlayerProfile, type PredictionResult, type Surface } from "../lib/model";
import { PlayerSearch, type SearchablePlayer } from "./PlayerSearch";

type ProductView = "match" | "bracket" | "live";
type MatchMode = "professional" | "custom";
type Side = "one" | "two";

const currentPlayers: SearchablePlayer[] = professionalPlayers.map((player) => ({
  ...player,
  surfaceRating: { ...player.surfaceRating },
  surfaceSamples: { ...player.surfaceSamples },
  careerStart: 2024,
  careerEnd: 2026,
  profileBasis: "current" as const,
}));

const surfaceLabels: Record<Surface, string> = { hard: "Hard", clay: "Clay", grass: "Grass" };
const percent = (value: number, digits = 0) => `${(value * 100).toFixed(digits)}%`;

const defaultAdvanced: Record<Side, AdvancedProfileInputs> = {
  one: { rating: 1850, surfaceRating: 1900, ratingUncertainty: 95, servePointsWon: 66, returnPointsWon: 39, formRate: 68, sampleMatches: 42, clutchIndex: 1.2, fitnessIndex: 1.5 },
  two: { rating: 1790, surfaceRating: 1810, ratingUncertainty: 110, servePointsWon: 63, returnPointsWon: 41, formRate: 61, sampleMatches: 31, clutchIndex: 0.4, fitnessIndex: 0.8 },
};

const advancedFields: Array<{ key: keyof AdvancedProfileInputs; label: string; unit: string; hint: string }> = [
  { key: "rating", label: "Overall rating", unit: "Elo", hint: "1500 is established-tour baseline; 2000+ is elite in this model." },
  { key: "surfaceRating", label: "Surface rating", unit: "Elo", hint: "Ability specifically on the selected court surface." },
  { key: "ratingUncertainty", label: "Rating uncertainty", unit: "σ", hint: "Lower means more evidence; 60 is known, 180 is highly uncertain." },
  { key: "servePointsWon", label: "Serve points won", unit: "%", hint: "Enter 64.5 or 0.645. Tour range is usually about 52–72%." },
  { key: "returnPointsWon", label: "Return points won", unit: "%", hint: "Enter 39 or 0.39. Higher means more pressure on opposing serve." },
  { key: "formRate", label: "Recent win rate", unit: "%", hint: "A recency signal, partially pooled by the simulator." },
  { key: "sampleMatches", label: "Evidence sample", unit: "matches", hint: "Controls how much the model trusts the entered rates." },
  { key: "clutchIndex", label: "Pressure index", unit: "open", hint: "0 is neutral; positive or negative values have no fixed UI cap." },
  { key: "fitnessIndex", label: "Endurance index", unit: "open", hint: "0 is neutral; affects later sets through a smooth response curve." },
];

function meaning(field: keyof AdvancedProfileInputs, value: number) {
  if (!Number.isFinite(value)) return "The model will substitute a neutral prior until a number is entered.";
  if (field === "rating") return `${value.toLocaleString()} places this profile ${value >= 2050 ? "among all-time elite priors" : value >= 1850 ? "at top-tour level" : value >= 1600 ? "at established professional level" : "in a developmental or lower-tour range"}.`;
  if (field === "surfaceRating") return `${value.toLocaleString()} is the latent strength used on the selected surface before uncertainty is sampled.`;
  if (field === "ratingUncertainty") return `A ±${Math.abs(value).toLocaleString()} rating-point standard deviation creates ${Math.abs(value) <= 75 ? "a tight" : Math.abs(value) <= 140 ? "a moderate" : "a wide"} posterior.`;
  if (field === "sampleMatches") return `${Math.max(0, value).toLocaleString()} matches supply the effective evidence volume; there is no form-field ceiling.`;
  if (field === "clutchIndex" || field === "fitnessIndex") return `${value > 0 ? "+" : ""}${value} is ${Math.abs(value) < 0.25 ? "effectively neutral" : value > 0 ? "a positive" : "a negative"} open-ended index, transformed smoothly so extreme entries stay numerically stable.`;
  const asPercent = Math.abs(value) <= 1 ? value * 100 : value;
  if (field === "servePointsWon") return `${asPercent.toFixed(1)}% implies ${asPercent >= 68 ? "dominant" : asPercent >= 63 ? "strong" : asPercent >= 57 ? "competitive" : "vulnerable"} service-point control.`;
  if (field === "returnPointsWon") return `${asPercent.toFixed(1)}% implies ${asPercent >= 43 ? "elite" : asPercent >= 39 ? "strong" : asPercent >= 35 ? "average" : "limited"} return pressure.`;
  return `${asPercent.toFixed(1)}% is treated as a recent-results signal, not a guaranteed future rate.`;
}

export function PredictionStudio() {
  const [view, setView] = useState<ProductView>("match");

  return <main>
    <header className="site-nav" id="top">
      <a className="brand" href="#top" aria-label="Baseline Labs home"><span className="brand-dot" /><span>BASELINE</span><b>LABS</b></a>
      <nav aria-label="Product navigation">
        <button className={view === "match" ? "active" : ""} onClick={() => setView("match")}>1v1 Lab</button>
        <button className={view === "bracket" ? "active" : ""} onClick={() => setView("bracket")}>Bracket Lab</button>
        <button className={view === "live" ? "active" : ""} onClick={() => setView("live")}><i /> Live Tour</button>
      </nav>
      <a className="method-link" href="#method">Model notes</a>
    </header>

    <section className="hero-clean">
      <div className="hero-kicker"><span>Probabilistic tennis intelligence</span><i /> Model v3.0</div>
      <div className="hero-grid">
        <h1>Forecast the point.<br />Then the path.</h1>
        <div><p>Compare any era, build an unrestricted player profile, simulate a full tournament, or follow a live draw as it changes.</p><button onClick={() => { setView("match"); document.getElementById("workspace")?.scrollIntoView({ behavior: "smooth" }); }}>Start forecasting <span>↘</span></button></div>
      </div>
      <div className="signal-strip">
        <div><strong>{playerDatabaseSummary.count.toLocaleString()}</strong><span>searchable ATP + WTA profiles</span></div>
        <div><strong>{playerDatabaseSummary.firstYear}—{playerDatabaseSummary.lastYear}</strong><span>historical coverage</span></div>
        <div><strong>5,040</strong><span>score-level simulations per 1v1</span></div>
        <div><strong>60 sec</strong><span>live bracket refresh</span></div>
      </div>
    </section>

    <section className="product-shell" id="workspace">
      <div className="workspace-tabs" role="tablist" aria-label="Forecast workspace">
        <button role="tab" aria-selected={view === "match"} className={view === "match" ? "active" : ""} onClick={() => setView("match")}><span>01</span><strong>1v1 Lab</strong><small>Any player, any era</small></button>
        <button role="tab" aria-selected={view === "bracket"} className={view === "bracket" ? "active" : ""} onClick={() => setView("bracket")}><span>02</span><strong>Bracket Lab</strong><small>Unlimited custom fields</small></button>
        <button role="tab" aria-selected={view === "live"} className={view === "live" ? "active" : ""} onClick={() => setView("live")}><span>03</span><strong>Live Tour</strong><small>Draw-aware forecasts</small></button>
      </div>
      {view === "match" ? <MatchLab /> : view === "bracket" ? <BracketLab /> : <LiveTourDesk />}
    </section>

    <Methodology />
    <footer>
      <a className="brand footer-brand" href="#top"><span className="brand-dot" /><span>BASELINE</span><b>LABS</b></a>
      <p>Transparent research software. Forecasts express model uncertainty and are not betting advice. No affiliation with the ATP, WTA, ESPN, or tournament operators.</p>
      <div><a href="https://www.usopen.org/en_US/about/eventschedule.html" target="_blank" rel="noreferrer">US Open schedule ↗</a><a href="https://github.com/Aneeshers/tennis-sackmann-archive" target="_blank" rel="noreferrer">Data archive ↗</a></div>
    </footer>
  </main>;
}

function MatchLab() {
  const [mode, setMode] = useState<MatchMode>("professional");
  const [one, setOne] = useState<SearchablePlayer>(currentPlayers[0]);
  const [two, setTwo] = useState<SearchablePlayer>(currentPlayers[1]);
  const [surface, setSurface] = useState<Surface>("hard");
  const [bestOf, setBestOf] = useState<3 | 5>(3);
  const [names, setNames] = useState<Record<Side, string>>({ one: "Player A", two: "Player B" });
  const [inputs, setInputs] = useState<Record<Side, AdvancedProfileInputs>>({ one: { ...defaultAdvanced.one }, two: { ...defaultAdvanced.two } });
  const [activeSide, setActiveSide] = useState<Side>("one");
  const [result, setResult] = useState<PredictionResult | null>(() => predictMatch(currentPlayers[0], currentPlayers[1], "hard", 3));
  const [running, setRunning] = useState(false);
  const customOne = useMemo(() => advancedProfile("advanced-one", names.one, inputs.one, surface), [inputs.one, names.one, surface]);
  const customTwo = useMemo(() => advancedProfile("advanced-two", names.two, inputs.two, surface), [inputs.two, names.two, surface]);
  const selectedOne = mode === "professional" ? one : customOne;
  const selectedTwo = mode === "professional" ? two : customTwo;

  function run() {
    setRunning(true);
    window.setTimeout(() => {
      setResult(predictMatch(selectedOne, selectedTwo, surface, bestOf));
      setRunning(false);
      window.setTimeout(() => document.getElementById("match-forecast")?.scrollIntoView({ behavior: "smooth", block: "center" }), 30);
    }, 60);
  }

  return <div className="lab-panel">
    <header className="panel-heading"><div><span>01 / HEAD TO HEAD</span><h2>Build one matchup.</h2></div><p>Use current or career-peak professional priors, or enter raw model inputs without a fixed scoring scale.</p></header>
    <div className="submode-switch">
      <button className={mode === "professional" ? "active" : ""} onClick={() => { setMode("professional"); setResult(null); }}><strong>Professional library</strong><small>Search every indexed era</small></button>
      <button className={mode === "custom" ? "active" : ""} onClick={() => { setMode("custom"); setResult(null); }}><strong>Unrestricted profile</strong><small>Direct statistical inputs</small></button>
    </div>

    {mode === "professional" ? <div className="versus-card">
      <PlayerSearch label="Player one" selected={one} excludeId={two.id} onSelect={(player) => { setOne(player); setResult(null); }} />
      <div className="versus-mark"><span>VS</span></div>
      <PlayerSearch label="Player two" selected={two} excludeId={one.id} onSelect={(player) => { setTwo(player); setResult(null); }} />
    </div> : <AdvancedBuilder active={activeSide} names={names} inputs={inputs} onActive={setActiveSide} onName={(side, name) => { setNames((current) => ({ ...current, [side]: name })); setResult(null); }} onInput={(side, key, value) => { setInputs((current) => ({ ...current, [side]: { ...current[side], [key]: value } })); setResult(null); }} />}

    <ForecastControls surface={surface} bestOf={bestOf} running={running} onSurface={(value) => { setSurface(value); setResult(null); }} onBestOf={(value) => { setBestOf(value); setResult(null); }} onRun={run} label="Run full forecast" />
    <ForecastPanel result={result} one={selectedOne} two={selectedTwo} surface={surface} />
  </div>;
}

function AdvancedBuilder({ active, names, inputs, onActive, onName, onInput }: {
  active: Side;
  names: Record<Side, string>;
  inputs: Record<Side, AdvancedProfileInputs>;
  onActive: (side: Side) => void;
  onName: (side: Side, name: string) => void;
  onInput: (side: Side, key: keyof AdvancedProfileInputs, value: number) => void;
}) {
  return <div className="advanced-builder">
    <div className="advanced-intro"><div><span>OPEN-ENDED INPUT MODEL</span><h3>No 1–10 ceiling.</h3></div><p>Enter raw ratings, percentages, evidence volume, and signed indices. Inputs accept decimals and values outside typical tour ranges; the probability engine keeps extreme values stable.</p></div>
    <div className="profile-tabs">{(["one", "two"] as Side[]).map((side, index) => <button key={side} className={active === side ? "active" : ""} onClick={() => onActive(side)}><span>PLAYER {index + 1}</span><strong>{names[side] || `Player ${index + 1}`}</strong><small>{Math.round(inputs[side].rating)} rating</small></button>)}</div>
    <div className="name-field"><label htmlFor="advanced-name">Player name</label><input id="advanced-name" value={names[active]} onChange={(event) => onName(active, event.target.value)} /><span>Numbers are interpreted below as you enter them.</span></div>
    <div className="advanced-grid">{advancedFields.map((field) => <label className="metric-field" key={field.key}>
      <span>{field.label}<small>{field.unit}</small></span>
      <input type="number" step="any" value={inputs[active][field.key]} onChange={(event) => onInput(active, field.key, Number(event.target.value))} />
      <em>{field.hint}</em>
      <p><b>Reading:</b> {meaning(field.key, inputs[active][field.key])}</p>
    </label>)}</div>
  </div>;
}

function ForecastControls({ surface, bestOf, running, onSurface, onBestOf, onRun, label }: {
  surface: Surface;
  bestOf: 3 | 5;
  running: boolean;
  onSurface: (surface: Surface) => void;
  onBestOf: (bestOf: 3 | 5) => void;
  onRun: () => void;
  label: string;
}) {
  return <div className="forecast-controls">
    <div><span className="control-label">Surface</span><div className="choice-row">{(["hard", "clay", "grass"] as Surface[]).map((item) => <button className={surface === item ? "active" : ""} key={item} onClick={() => onSurface(item)}><i className={item} />{surfaceLabels[item]}</button>)}</div></div>
    <div><span className="control-label">Match format</span><div className="choice-row">{([3, 5] as const).map((item) => <button className={bestOf === item ? "active" : ""} key={item} onClick={() => onBestOf(item)}>Best of {item}</button>)}</div></div>
    <button className="primary-action" onClick={onRun} disabled={running}>{running ? "Simulating…" : label}<span>{running ? "◌" : "↗"}</span></button>
  </div>;
}

function ForecastPanel({ result, one, two, surface }: { result: PredictionResult | null; one: PlayerProfile; two: PlayerProfile; surface: Surface }) {
  if (!result) return <div className="forecast-waiting" id="match-forecast"><span>MODEL OUTPUT</span><strong>Ready when you are.</strong><p>Run the forecast to sample latent ability and play the score point by point.</p></div>;
  const winnerIsOne = result.projectedWinner === one.name;
  const winProbability = winnerIsOne ? result.playerOneProbability : result.playerTwoProbability;
  const low = winnerIsOne ? result.intervalLow : 1 - result.intervalHigh;
  const high = winnerIsOne ? result.intervalHigh : 1 - result.intervalLow;
  return <section className="forecast-result" id="match-forecast">
    <div className="result-hero"><div><span>PROJECTED WINNER · {surfaceLabels[surface].toUpperCase()}</span><h3>{result.projectedWinner}</h3><p>{percent(low)}–{percent(high)} posterior interval · {result.confidence.toLowerCase()} evidence</p></div><strong>{percent(winProbability)}</strong></div>
    <div className="split-meter"><i style={{ width: `${result.playerOneProbability * 100}%` }} /><span>{one.name} · {percent(result.playerOneProbability)}</span><span>{percent(result.playerTwoProbability)} · {two.name}</span></div>
    <div className="result-stats"><div><span>Likely score</span><strong>{result.likelySetScore}</strong></div><div><span>Expected sets</span><strong>{result.expectedSets.toFixed(2)}</strong></div><div><span>Tiebreak chance</span><strong>{percent(result.tieBreakChance)}</strong></div><div><span>Expected games</span><strong>{result.expectedGames.toFixed(1)}</strong></div></div>
    <div className="evidence-clean"><header><span>WHY THE MODEL LEANS</span><p>Separate diagnostics, not checklist weights.</p></header>{result.evidence.map((item) => <div key={item.label}><span><strong>{item.label}</strong><small>{item.detail}</small></span><i><b style={{ width: `${Math.max(6, item.strength * 100)}%` }} /></i><em>{item.leader === "one" ? one.name.split(" ").at(-1) : item.leader === "two" ? two.name.split(" ").at(-1) : "Even"}</em></div>)}</div>
  </section>;
}

function editableParticipant(id: string, name: string, rating: number, surface: Surface): BracketParticipant {
  const base: AdvancedProfileInputs = { ...defaultAdvanced.one, rating, surfaceRating: rating, ratingUncertainty: 135, sampleMatches: 18, formRate: 50, clutchIndex: 0, fitnessIndex: 0 };
  return { id, name, profile: advancedProfile(id, name, base, surface) };
}

function entrantId(prefix: string) {
  return `${prefix}-${globalThis.crypto.randomUUID()}`;
}

function BracketLab() {
  const [surface, setSurface] = useState<Surface>("hard");
  const [bestOf, setBestOf] = useState<3 | 5>(3);
  const [category, setCategory] = useState("Tour 1000");
  const [participants, setParticipants] = useState<BracketParticipant[]>(() => currentPlayers.slice(0, 8).map((player) => ({ id: player.id, name: player.name, profile: player })));
  const [newName, setNewName] = useState("");
  const [pasteOpen, setPasteOpen] = useState(false);
  const [rosterText, setRosterText] = useState("");
  const [result, setResult] = useState<ForecastBracket | null>(null);
  const [running, setRunning] = useState(false);

  function addPlayer(player: SearchablePlayer) {
    setParticipants((current) => [...current, { id: entrantId(player.id), name: player.name, profile: player }]);
    setResult(null);
  }

  function addNamed() {
    const name = newName.trim() || `Entrant ${participants.length + 1}`;
    const id = entrantId("entrant");
    setParticipants((current) => [...current, editableParticipant(id, name, 1500, surface)]);
    setNewName("");
    setResult(null);
  }

  function importRoster() {
    const names = rosterText.split(/\n|,/).map((name) => name.trim()).filter(Boolean);
    if (!names.length) return;
    setParticipants(names.map((name) => editableParticipant(entrantId("import"), name, 1500, surface)));
    setRosterText("");
    setPasteOpen(false);
    setResult(null);
  }

  function updateParticipant(index: number, key: "name" | "rating", value: string | number) {
    setParticipants((current) => current.map((participant, itemIndex) => {
      if (itemIndex !== index) return participant;
      if (key === "name") return { ...participant, name: String(value), profile: { ...participant.profile, name: String(value) } };
      const rating = Number(value);
      const delta = rating - participant.profile.rating;
      return { ...participant, profile: { ...participant.profile, rating, surfaceRating: { hard: participant.profile.surfaceRating.hard + delta, clay: participant.profile.surfaceRating.clay + delta, grass: participant.profile.surfaceRating.grass + delta } } };
    }));
    setResult(null);
  }

  function runBracket() {
    if (participants.length < 2) return;
    setRunning(true);
    window.setTimeout(() => { setResult(forecastBracket(participants, surface, bestOf)); setRunning(false); }, 40);
  }

  return <div className="lab-panel">
    <header className="panel-heading"><div><span>02 / TOURNAMENT SIMULATOR</span><h2>Build the whole draw.</h2></div><p>Add two players or two hundred. Non-power-of-two fields receive automatic byes; every played matchup gets its own model probability and likely score.</p></header>
    <div className="bracket-layout">
      <aside className="roster-panel">
        <div className="roster-head"><div><span>FIELD</span><strong>{participants.length} entrants</strong></div><small>No fixed participant cap</small></div>
        <PlayerSearch compact label="Add professional" onSelect={addPlayer} />
        <div className="quick-add"><input value={newName} onChange={(event) => setNewName(event.target.value)} onKeyDown={(event) => { if (event.key === "Enter") addNamed(); }} placeholder="Add a custom entrant" /><button onClick={addNamed}>Add</button></div>
        <button className="paste-toggle" onClick={() => setPasteOpen((current) => !current)}>{pasteOpen ? "Close roster import" : "Paste a full roster"} <span>↘</span></button>
        {pasteOpen ? <div className="roster-import"><textarea value={rosterText} onChange={(event) => setRosterText(event.target.value)} placeholder={'One player per line\nRoger Federer\nSerena Williams\n...'} /><button onClick={importRoster}>Replace field from list</button></div> : null}
        <div className="roster-list">{participants.map((participant, index) => <div key={participant.id} className="roster-row"><span>{String(index + 1).padStart(2, "0")}</span><input value={participant.name} onChange={(event) => updateParticipant(index, "name", event.target.value)} aria-label={`Entrant ${index + 1} name`} /><label><input type="number" step="any" value={Math.round(participant.profile.rating)} onChange={(event) => updateParticipant(index, "rating", event.target.value)} aria-label={`${participant.name} rating`} /><small>Elo</small></label><button onClick={() => { setParticipants((current) => current.filter((_, itemIndex) => itemIndex !== index)); setResult(null); }} aria-label={`Remove ${participant.name}`}>×</button></div>)}</div>
      </aside>
      <div className="tournament-setup">
        <div className="setup-grid"><label>Tournament type<select value={category} onChange={(event) => { const value = event.target.value; setCategory(value); if (value === "Grand Slam") setBestOf(5); else setBestOf(3); setResult(null); }}><option>Grand Slam</option><option>Tour 1000</option><option>Tour 500</option><option>Tour 250</option><option>Exhibition</option><option>Custom event</option></select></label><div className="setup-label">Draw behavior<strong>Single elimination + automatic byes</strong></div></div>
        <ForecastControls surface={surface} bestOf={bestOf} running={running} onSurface={(value) => { setSurface(value); setResult(null); }} onBestOf={(value) => { setBestOf(value); setResult(null); }} onRun={runBracket} label={`Predict ${participants.length}-player bracket`} />
        {result ? <PredictedBracket bracket={result} category={category} /> : <div className="bracket-empty"><span>BRACKET PREVIEW</span><h3>Your predicted path will appear here.</h3><p>Entrants are paired in listed draw order. Add, remove, paste, or edit as many entries as you need.</p></div>}
      </div>
    </div>
  </div>;
}

function PredictedBracket({ bracket, category }: { bracket: ForecastBracket; category: string }) {
  return <section className="predicted-bracket">
    <header><div><span>PREDICTED {category.toUpperCase()} DRAW</span><h3>{bracket.champion?.name ?? "No champion"}</h3><p>projected champion · {bracket.size}-slot draw</p></div><div><strong>{bracket.rounds.length}</strong><span>rounds modeled</span></div></header>
    <div className="bracket-scroll">{bracket.rounds.map((round) => <section className="round-column" key={round.label}><h4>{round.label}<span>{round.matches.length}</span></h4>{round.matches.map((match) => <article className="bracket-match" key={match.id}>
      <div className={match.winner?.id === match.one?.id ? "winner" : ""}><span>{match.one?.name ?? "BYE"}</span><b>{match.firstProbability === null ? "—" : percent(match.firstProbability)}</b></div>
      <div className={match.winner?.id === match.two?.id ? "winner" : ""}><span>{match.two?.name ?? "BYE"}</span><b>{match.firstProbability === null ? "—" : percent(1 - match.firstProbability)}</b></div>
      <footer><span>{match.bye ? "Automatic advance" : `Projected ${match.score}`}</span><strong>{match.winner?.name ?? "Open"}</strong></footer>
    </article>)}</section>)}</div>
  </section>;
}

interface LiveMatch {
  id: string;
  round: string;
  roundId: number;
  startsAt: string | null;
  court: string | null;
  state: string;
  status: string;
  players: Array<{ name: string; winner: boolean; score: string }>;
  forecast: null | { winner: string; firstProbability: number; score: string; confidence: string };
}

interface LiveTournament {
  id: string;
  tour: string;
  name: string;
  venue: string;
  surface: string;
  bracketLink: string | null;
  matches: LiveMatch[];
}

function roundOrder(label: string) {
  const order = ["Qualifying 1st Round", "Qualifying 2nd Round", "Qualifying", "Round 1", "Round 2", "Round 3", "Round 4", "Quarterfinal", "Semifinal", "Final"];
  const index = order.findIndex((item) => label.includes(item));
  return index < 0 ? 99 : index;
}

function LiveTourDesk() {
  const [tournaments, setTournaments] = useState<LiveTournament[]>([]);
  const [selectedId, setSelectedId] = useState("");
  const [updatedAt, setUpdatedAt] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  useEffect(() => {
    let active = true;
    async function refresh() {
      try {
        const response = await fetch("/api/live", { cache: "no-store" });
        const data = await response.json() as { tournaments?: LiveTournament[]; updatedAt?: string; error?: string };
        if (!response.ok) throw new Error(data.error || "Live draw unavailable");
        if (!active) return;
        setTournaments(data.tournaments ?? []);
        setSelectedId((current) => current && data.tournaments?.some((event) => event.id === current) ? current : data.tournaments?.[0]?.id ?? "");
        setUpdatedAt(data.updatedAt ?? new Date().toISOString());
        setError("");
      } catch (caught) {
        if (active) setError(caught instanceof Error ? caught.message : "Live draw unavailable");
      } finally {
        if (active) setLoading(false);
      }
    }
    refresh();
    const timer = window.setInterval(refresh, 60_000);
    return () => { active = false; window.clearInterval(timer); };
  }, []);

  const selected = tournaments.find((event) => event.id === selectedId) ?? tournaments[0];
  const rounds = useMemo(() => {
    if (!selected) return [];
    const grouped = new Map<string, LiveMatch[]>();
    for (const match of selected.matches) grouped.set(match.round, [...(grouped.get(match.round) ?? []), match]);
    return [...grouped.entries()].sort((a, b) => roundOrder(a[0]) - roundOrder(b[0]));
  }, [selected]);

  return <div className="lab-panel live-panel">
    <header className="panel-heading"><div><span>03 / LIVE TOUR DESK</span><h2>The draw, still moving.</h2></div><p>Completed results lock into the bracket. Known future matchups are re-forecast from the latest field every sixty seconds.</p></header>
    <div className="live-status"><div><i className={error ? "error" : ""} /><strong>{error ? "Feed interrupted" : "Auto-refreshing"}</strong><span>{updatedAt ? `Last checked ${new Date(updatedAt).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}` : "Connecting to tour scoreboards"}</span></div><span>LIVE RESULTS + MODEL LAYER</span></div>
    {loading ? <div className="live-loading"><i /><strong>Reading today’s ATP and WTA draws…</strong></div> : error && !selected ? <div className="live-loading error"><strong>Live data is temporarily unavailable.</strong><p>{error}</p></div> : selected ? <>
      <div className="event-tabs">{tournaments.map((event) => <button key={event.id} className={selected.id === event.id ? "active" : ""} onClick={() => setSelectedId(event.id)}><span>{event.tour}</span><strong>{event.name}</strong><small>{event.venue}</small></button>)}</div>
      <div className="live-event-head"><div><span>{selected.tour} · {selected.surface.toUpperCase()} COURT</span><h3>{selected.name}</h3><p>{selected.venue} · actual results are preserved; unplayed matches show model scores and probabilities.</p></div>{selected.bracketLink ? <a href={selected.bracketLink} target="_blank" rel="noreferrer">Official draw ↗</a> : null}</div>
      <div className="live-bracket-scroll">{rounds.map(([round, matches]) => <section className="live-round" key={round}><h4>{round}<span>{matches.length}</span></h4>{matches.map((match) => <article className={`live-match ${match.state}`} key={match.id}>
        <header><span>{match.state === "post" ? "FINAL" : match.state === "in" ? "IN PLAY" : "FORECAST"}</span><small>{match.court || match.status}</small></header>
        {match.players.map((player, index) => <div className={player.winner || match.forecast?.winner === player.name ? "winner" : ""} key={`${match.id}-${player.name}-${index}`}><span>{player.name}</span><b>{match.state === "post" ? player.score : match.forecast ? percent(index === 0 ? match.forecast.firstProbability : 1 - match.forecast.firstProbability) : "—"}</b></div>)}
        <footer>{match.state === "post" ? <span>Actual result</span> : match.forecast ? <><span>Projected {match.forecast.score}</span><strong>{match.forecast.winner}</strong></> : <span>Awaiting prior-round winner</span>}</footer>
      </article>)}</section>)}</div>
      <div className="live-disclosure"><strong>How updates work</strong><p>The scoreboard is requested every 60 seconds. Finished matches replace forecasts; newly resolved matchups receive fresh posterior simulations. Timing can trail the official tournament feed.</p></div>
    </> : <div className="live-loading"><strong>No ATP or WTA singles tournament was returned for today.</strong></div>}
  </div>;
}

function Methodology() {
  return <section className="methodology-clean" id="method">
    <header><div><span>MODEL / TRANSPARENCY</span><h2>Serious mechanics.<br />Visible limits.</h2></div><p>The forecast is a dynamic paired-comparison model with surface pooling, Bayesian skill uncertainty, and literal tennis scoring—not a weighted checklist.</p></header>
    <div className="method-cards"><article><span>01</span><h3>Latent ability</h3><p>Opponent-adjusted ratings evolve through match history. Retired-player comparisons use a career-peak prior; active players use the latest archive state.</p></article><article><span>02</span><h3>Surface transfer</h3><p>Hard, clay, and grass evidence is partially pooled toward overall strength when samples are small.</p></article><article><span>03</span><h3>Posterior skills</h3><p>Serve, return, and latent rating are resampled so the output includes model uncertainty instead of false precision.</p></article><article><span>04</span><h3>Score engine</h3><p>Points become advantage games, tiebreaks, sets, matches, then bracket paths. Byes advance without invented play.</p></article></div>
    <div className="method-note"><strong>Responsible use</strong><p>Live injuries, travel, coaching changes, weather, and same-day news are not fully represented. Cross-era comparisons are counterfactual. Treat probabilities as research estimates, never guarantees.</p><a href="https://arxiv.org/abs/1902.07378" target="_blank" rel="noreferrer">Research basis ↗</a></div>
  </section>;
}

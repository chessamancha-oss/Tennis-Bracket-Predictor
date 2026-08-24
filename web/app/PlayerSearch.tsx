"use client";

import { useEffect, useId, useState } from "react";
import type { CataloguePlayer } from "../lib/player-database";
import type { PlayerProfile, Tour } from "../lib/model";

export type SearchablePlayer = PlayerProfile & Partial<Pick<CataloguePlayer, "careerStart" | "careerEnd" | "careerMatches" | "careerWins" | "majorTitles" | "profileBasis">>;

const eras = [
  { label: "All eras", start: undefined, end: undefined },
  { label: "Active", start: 2025, end: 2026 },
  { label: "2010s–20s", start: 2010, end: 2026 },
  { label: "1990s–00s", start: 1990, end: 2009 },
  { label: "1970s–80s", start: 1970, end: 1989 },
  { label: "Early era", start: 1900, end: 1969 },
] as const;

function era(player: SearchablePlayer) {
  if (player.careerStart && player.careerEnd) return `${player.careerStart}–${player.careerEnd}`;
  return player.rank ? `World No. ${player.rank}` : "Tour profile";
}
export function PlayerSearch({ label, selected, onSelect, excludeId, compact = false }: {
  label: string;
  selected?: SearchablePlayer | null;
  onSelect: (player: SearchablePlayer) => void;
  excludeId?: string;
  compact?: boolean;
}) {
  const id = useId();
  const [query, setQuery] = useState("");
  const [tour, setTour] = useState<"ALL" | Tour>("ALL");
  const [eraIndex, setEraIndex] = useState(0);
  const [players, setPlayers] = useState<SearchablePlayer[]>([]);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (!open) return;
    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      setLoading(true);
      const selectedEra = eras[eraIndex];
      const params = new URLSearchParams({ q: query });
      if (tour !== "ALL") params.set("tour", tour);
      if (selectedEra.start) params.set("eraStart", String(selectedEra.start));
      if (selectedEra.end) params.set("eraEnd", String(selectedEra.end));
      try {
        const response = await fetch(`/api/players?${params}`, { signal: controller.signal });
        const payload = await response.json() as { players?: SearchablePlayer[] };
        setPlayers((payload.players ?? []).filter((player) => player.id !== excludeId));
      } catch (error) {
        if ((error as Error).name !== "AbortError") setPlayers([]);
      } finally {
        setLoading(false);
      }
    }, 180);
    return () => { window.clearTimeout(timer); controller.abort(); };
  }, [eraIndex, excludeId, open, query, tour]);

  function choose(player: SearchablePlayer) {
    onSelect(player);
    setQuery("");
    setOpen(false);
  }

  return <div className={`player-search ${compact ? "compact" : ""}`}>
    <label htmlFor={id}>{label}</label>
    {selected && !compact ? <div className="selected-player">
      <div><span>{selected.tour} · {era(selected)}</span><strong>{selected.name}</strong><small>{selected.country} · {selected.hand}-handed</small></div>
      <div><b>{Math.round(selected.rating)}</b><span>model rating</span></div>
    </div> : null}
    <div className="search-control">
      <span aria-hidden="true">⌕</span>
      <input id={id} value={query} onFocus={() => setOpen(true)} onChange={(event) => { setQuery(event.target.value); setOpen(true); }} placeholder={compact ? "Search 7,255 players…" : "Search any current or historical player…"} autoComplete="off" />
      {open ? <button type="button" onClick={() => setOpen(false)} aria-label="Close player search">×</button> : null}
    </div>
    {open ? <div className="search-popover">
      <div className="search-filters">
        <div>{(["ALL", "ATP", "WTA"] as const).map((item) => <button type="button" key={item} className={tour === item ? "active" : ""} onClick={() => setTour(item)}>{item}</button>)}</div>
        <select aria-label="Player era" value={eraIndex} onChange={(event) => setEraIndex(Number(event.target.value))}>{eras.map((item, index) => <option value={index} key={item.label}>{item.label}</option>)}</select>
      </div>
      <div className="search-results" role="listbox" aria-label={`${label} results`}>
        {loading ? <p className="search-state">Searching the archive…</p> : players.length ? players.map((player) => <button type="button" role="option" aria-selected={false} key={player.id} onClick={() => choose(player)}>
          <span className="result-monogram">{player.name.split(" ").slice(0, 2).map((part) => part[0]).join("")}</span>
          <span><strong>{player.name}</strong><small>{player.tour} · {player.country} · {era(player)}</small></span>
          <span><b>{Math.round(player.rating)}</b><small>{player.majorTitles ? `${player.majorTitles} majors` : player.rank ? `No. ${player.rank}` : "peak"}</small></span>
        </button>) : <p className="search-state">No player found. Try a surname or a different era.</p>}
      </div>
      <div className="search-source"><i /> Career-peak profiles for retired players · current priors for active players</div>
    </div> : null}
  </div>;
}

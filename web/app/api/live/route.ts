import { NextResponse } from "next/server";
import { catalogueByNames, fallbackTourProfile, normalizePlayerName } from "../../../lib/player-database";
import { predictMatch, type PlayerProfile, type Tour } from "../../../lib/model";
import { emptyTournamentAccuracy, recordAndGradeLivePredictions } from "../../../lib/prediction-ledger";

interface RawCompetitor {
  winner?: boolean;
  curatedRank?: { current?: number };
  athlete?: { displayName?: string; flag?: { alt?: string } };
  linescores?: Array<{ value?: number; tiebreak?: number }>;
}

interface RawCompetition {
  id?: string;
  date?: string;
  status?: { type?: { state?: string; description?: string; shortDetail?: string } };
  competitors?: RawCompetitor[];
  round?: { id?: string; displayName?: string };
  venue?: { court?: string };
}

interface RawGrouping {
  grouping?: { slug?: string; displayName?: string };
  competitions?: RawCompetition[];
}

interface RawEvent {
  id?: string;
  name?: string;
  date?: string;
  endDate?: string;
  venue?: { displayName?: string };
  groupings?: RawGrouping[];
  links?: Array<{ rel?: string[]; href?: string }>;
}

interface Scoreboard {
  events?: RawEvent[];
}

function isoDay(date: Date) {
  return date.toISOString().slice(0, 10).replaceAll("-", "");
}

function competitorName(competitor: RawCompetitor) {
  return competitor.athlete?.displayName?.trim() || "TBD";
}

function score(competitor: RawCompetitor) {
  return (competitor.linescores ?? []).map((line) => {
    const value = line.value ?? 0;
    return line.tiebreak !== undefined ? `${value}(${line.tiebreak})` : String(value);
  }).join(" ");
}

function inferSurface(event: RawEvent): "hard" | "clay" | "grass" {
  const description = `${event.name ?? ""} ${event.venue?.displayName ?? ""}`.toLowerCase();
  const grassEvents = ["wimbledon", "queen's", "queens", "halle", "eastbourne", "nottingham", "bad homburg", "mallorca", "s-hertogenbosch"];
  const clayEvents = ["roland garros", "french open", "monte-carlo", "monte carlo", "madrid", "rome", "internazionali bnl", "barcelona", "hamburg", "munich", "bastad", "gstaad", "umag", "kitzbuhel", "kitzbühel", "stuttgart"];
  if (grassEvents.some((name) => description.includes(name))) return "grass";
  if (clayEvents.some((name) => description.includes(name))) return "clay";
  return "hard";
}

async function tournament(event: RawEvent, tour: Tour) {
  const desiredGrouping = tour === "ATP" ? "mens-singles" : "womens-singles";
  const singles = (event.groupings ?? []).find((group) => group.grouping?.slug === desiredGrouping);
  if (!singles?.competitions?.length) return null;
  const names = singles.competitions.flatMap((match) => (match.competitors ?? []).map(competitorName)).filter((name) => name !== "TBD");
  let catalogue = new Map<string, PlayerProfile>();
  try {
    catalogue = await catalogueByNames(names);
  } catch {
    // A live scoreboard remains useful while the local/deployed catalogue initializes.
  }
  const surface = inferSurface(event);
  const matches = singles.competitions.map((match) => {
    const competitors = [...(match.competitors ?? [])].sort((a, b) => Number(Boolean(b.winner)) - Number(Boolean(a.winner)));
    const first = competitors[0];
    const second = competitors[1];
    const firstName = first ? competitorName(first) : "TBD";
    const secondName = second ? competitorName(second) : "TBD";
    const state = match.status?.type?.state ?? "pre";
    let forecast = null;
    if (state !== "post" && firstName !== "TBD" && secondName !== "TBD") {
      const firstProfile = catalogue.get(normalizePlayerName(firstName)) ?? fallbackTourProfile(firstName, tour, first?.curatedRank?.current);
      const secondProfile = catalogue.get(normalizePlayerName(secondName)) ?? fallbackTourProfile(secondName, tour, second?.curatedRank?.current);
      const result = predictMatch(firstProfile, secondProfile, surface, 3, 540, 18);
      forecast = {
        winner: result.projectedWinner,
        firstProbability: result.playerOneProbability,
        score: result.likelySetScore,
        confidence: result.confidence,
      };
    }
    return {
      id: match.id ?? `${firstName}-${secondName}`,
      round: match.round?.displayName ?? "Draw",
      roundId: Number(match.round?.id ?? 0),
      startsAt: match.date ?? null,
      court: match.venue?.court || null,
      state,
      status: match.status?.type?.shortDetail ?? match.status?.type?.description ?? "Scheduled",
      players: [
        { name: firstName, winner: Boolean(first?.winner), score: first ? score(first) : "" },
        { name: secondName, winner: Boolean(second?.winner), score: second ? score(second) : "" },
      ],
      forecast,
    };
  });
  const bracketLink = event.links?.find((link) => link.rel?.includes("bracket"))?.href ?? null;
  return {
    id: `${tour}-${event.id ?? event.name}`,
    tour,
    name: event.name ?? `${tour} tournament`,
    venue: event.venue?.displayName ?? "Tour event",
    startsAt: event.date ?? null,
    endsAt: event.endDate ?? null,
    surface: surface[0].toUpperCase() + surface.slice(1),
    bracketLink,
    matches,
  };
}

function tournamentSummary(event: RawEvent, tour: Tour) {
  const desiredGrouping = tour === "ATP" ? "mens-singles" : "womens-singles";
  const singles = (event.groupings ?? []).find((group) => group.grouping?.slug === desiredGrouping);
  if (!singles?.competitions?.length) return null;
  const surface = inferSurface(event);
  return {
    id: `${tour}-${event.id ?? event.name}`,
    tour,
    name: event.name ?? `${tour} tournament`,
    venue: event.venue?.displayName ?? "Tour event",
    surface,
    startsAt: event.date ?? null,
    endsAt: event.endDate ?? null,
  };
}

export async function GET(request: Request) {
  const now = new Date();
  const date = isoDay(now);
  const summaryOnly = new URL(request.url).searchParams.get("summary") === "1";
  try {
    const tours: Tour[] = ["ATP", "WTA"];
    const scoreboards: Array<readonly [Tour, Scoreboard | null]> = [];
    for (const tour of tours) {
      const response = await fetch(`https://site.api.espn.com/apis/site/v2/sports/tennis/${tour.toLowerCase()}/scoreboard?dates=${date}`, {
        headers: {
          Accept: "application/json",
          "User-Agent": "curl/8.7.1",
        },
        cache: "no-store",
      });
      scoreboards.push([tour, response.ok ? await response.json() as Scoreboard : null] as const);
    }
    if (summaryOnly) {
      const tournaments = scoreboards.flatMap(([tour, board]) => (board?.events ?? []).map((event) => tournamentSummary(event, tour))).filter((item) => item !== null);
      return NextResponse.json({ tournaments, updatedAt: now.toISOString(), refreshSeconds: 300 }, {
        headers: { "Cache-Control": "public, max-age=120, s-maxage=300, stale-while-revalidate=900" },
      });
    }
    const nested = await Promise.all(scoreboards.flatMap(([tour, board]) => (board?.events ?? []).map((event) => tournament(event, tour))));
    const tournaments = nested.filter((item) => item !== null);
    let scorecardAvailable = true;
    let accuracyByTournament = new Map();
    try {
      accuracyByTournament = await recordAndGradeLivePredictions(tournaments, now.toISOString());
    } catch {
      scorecardAvailable = false;
    }
    const tournamentsWithAccuracy = tournaments.map((event) => ({
      ...event,
      accuracy: accuracyByTournament.get(event.id) ?? emptyTournamentAccuracy(),
    }));
    return NextResponse.json({ tournaments: tournamentsWithAccuracy, updatedAt: now.toISOString(), refreshSeconds: 60, source: "Live tour scoreboard", scorecardAvailable }, {
      headers: { "Cache-Control": "public, max-age=30, s-maxage=45, stale-while-revalidate=120" },
    });
  } catch (error) {
    return NextResponse.json({ tournaments: [], updatedAt: now.toISOString(), refreshSeconds: 60, error: error instanceof Error ? error.message : "Live data unavailable" }, { status: 502 });
  }
}

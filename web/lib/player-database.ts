import { env } from "cloudflare:workers";
import type { PlayerProfile, Tour } from "./model";

export interface CataloguePlayer extends PlayerProfile {
  careerStart: number;
  careerEnd: number;
  careerMatches: number;
  careerWins: number;
  majorTitles: number;
  profileBasis: "current" | "career-peak";
}
interface PlayerRow {
  id: string;
  tour: Tour;
  name: string;
  search_key: string;
  country: string;
  hand: string;
  birth_year: number | null;
  career_start: number;
  career_end: number;
  rank: number | null;
  ranking_points: number | null;
  rating: number;
  rating_sigma: number;
  matches: number;
  wins: number;
  form_rate: number;
  serve_points_won: number;
  return_points_won: number;
  hold_rate: number;
  ace_rate: number | null;
  double_fault_rate: number | null;
  serve_sample: number;
  return_sample: number;
  hard_rating: number;
  hard_matches: number;
  clay_rating: number;
  clay_matches: number;
  grass_rating: number;
  grass_matches: number;
  major_titles: number;
  last_match_date: string | null;
}

interface DatabaseEnv {
  DB?: D1Database;
}

export function normalizePlayerName(value: string) {
  return value.normalize("NFKD").replace(/[\u0300-\u036f]/g, "").toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
}

export function fallbackTourProfile(name: string, tour: Tour, seed?: number): PlayerProfile {
  const rating = seed ? 1880 - Math.log2(Math.max(1, seed)) * 52 : 1510;
  const serve = tour === "ATP" ? 0.635 : 0.585;
  return {
    id: `live-${tour.toLowerCase()}-${normalizePlayerName(name).replaceAll(" ", "-")}`,
    tour,
    name,
    rank: null,
    rankingPoints: null,
    country: "Live draw",
    hand: "Unknown",
    age: null,
    rating,
    surfaceRating: { hard: rating, clay: rating - 8, grass: rating - 4 },
    ratingSigma: seed ? 118 : 172,
    matches52w: seed ? 22 : 6,
    wins52w: seed ? 14 : 3,
    form90d: seed ? 0.61 : 0.5,
    servePointsWon: serve,
    returnPointsWon: 1 - serve,
    holdRate: tour === "ATP" ? 0.79 : 0.72,
    aceRate: null,
    doubleFaultRate: null,
    serveSample: 30,
    returnSample: 30,
    surfaceSamples: { hard: 4, clay: 4, grass: 2 },
    lastMatchDate: null,
    rankingSnapshot: null,
    historyCutoff: null,
  };
}

function database() {
  return (env as unknown as DatabaseEnv).DB;
}

function rowToProfile(row: PlayerRow): CataloguePlayer {
  const sample = Math.min(80, row.matches);
  return {
    id: row.id,
    tour: row.tour,
    name: row.name,
    rank: row.rank,
    rankingPoints: row.ranking_points,
    country: row.country,
    hand: row.hand,
    age: row.birth_year ? Math.max(0, row.career_end - row.birth_year) : null,
    rating: row.rating,
    surfaceRating: { hard: row.hard_rating, clay: row.clay_rating, grass: row.grass_rating },
    ratingSigma: row.rating_sigma,
    matches52w: sample,
    wins52w: Math.round(sample * row.form_rate),
    form90d: row.form_rate,
    servePointsWon: row.serve_points_won,
    returnPointsWon: row.return_points_won,
    holdRate: row.hold_rate,
    aceRate: row.ace_rate,
    doubleFaultRate: row.double_fault_rate,
    serveSample: row.serve_sample,
    returnSample: row.return_sample,
    surfaceSamples: { hard: row.hard_matches, clay: row.clay_matches, grass: row.grass_matches },
    lastMatchDate: row.last_match_date,
    rankingSnapshot: row.rank ? "2026 archive snapshot" : null,
    historyCutoff: row.last_match_date,
    careerStart: row.career_start,
    careerEnd: row.career_end,
    careerMatches: row.matches,
    careerWins: row.wins,
    majorTitles: row.major_titles,
    profileBasis: row.career_end >= 2025 ? "current" : "career-peak",
  };
}

export async function searchCatalogue(options: {
  query?: string;
  tour?: Tour;
  eraStart?: number;
  eraEnd?: number;
  limit?: number;
}): Promise<CataloguePlayer[]> {
  const db = database();
  if (!db) return [];
  const conditions: string[] = [];
  const bindings: Array<string | number> = [];
  const query = normalizePlayerName(options.query ?? "");
  if (query) {
    conditions.push("search_key LIKE ?");
    bindings.push(`%${query}%`);
  }
  if (options.tour) {
    conditions.push("tour = ?");
    bindings.push(options.tour);
  }
  if (options.eraStart !== undefined) {
    conditions.push("career_end >= ?");
    bindings.push(options.eraStart);
  }
  if (options.eraEnd !== undefined) {
    conditions.push("career_start <= ?");
    bindings.push(options.eraEnd);
  }
  const where = conditions.length ? `WHERE ${conditions.join(" AND ")}` : "";
  const limit = Math.max(1, Math.min(60, Math.floor(options.limit ?? 24)));
  const statement = db.prepare(`SELECT * FROM players ${where} ORDER BY rating DESC, name ASC LIMIT ?`).bind(...bindings, limit);
  const result = await statement.all<PlayerRow>();
  return result.results.map(rowToProfile);
}

export async function catalogueByNames(names: string[]): Promise<Map<string, CataloguePlayer>> {
  const db = database();
  const keys = [...new Set(names.map(normalizePlayerName).filter(Boolean))];
  if (!db || keys.length === 0) return new Map();
  const placeholders = keys.map(() => "?").join(",");
  const result = await db.prepare(`SELECT * FROM players WHERE search_key IN (${placeholders}) ORDER BY rating DESC`).bind(...keys).all<PlayerRow>();
  return new Map(result.results.map((row) => [row.search_key, rowToProfile(row)]));
}

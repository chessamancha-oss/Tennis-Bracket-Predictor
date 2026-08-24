import { NextResponse } from "next/server";
import { professionalPlayers } from "../../../data/players.generated";
import { playerDatabaseSummary } from "../../../data/player-database-summary.generated";
import { searchCatalogue } from "../../../lib/player-database";
import type { PlayerProfile, Tour } from "../../../lib/model";

function currentFallback(query: string, tour?: Tour): PlayerProfile[] {
  const normalized = query.toLowerCase().trim();
  return professionalPlayers
    .filter((player) => (!tour || player.tour === tour) && (!normalized || player.name.toLowerCase().includes(normalized)))
    .slice(0, 24)
    .map((player) => ({ ...player, surfaceRating: { ...player.surfaceRating }, surfaceSamples: { ...player.surfaceSamples } }));
}
export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const query = searchParams.get("q")?.slice(0, 80) ?? "";
  const rawTour = searchParams.get("tour");
  const tour = rawTour === "ATP" || rawTour === "WTA" ? rawTour : undefined;
  const eraStart = Number(searchParams.get("eraStart"));
  const eraEnd = Number(searchParams.get("eraEnd"));
  try {
    const players = await searchCatalogue({
      query,
      tour,
      eraStart: Number.isFinite(eraStart) && eraStart > 0 ? eraStart : undefined,
      eraEnd: Number.isFinite(eraEnd) && eraEnd > 0 ? eraEnd : undefined,
      limit: 30,
    });
    return NextResponse.json({ players: players.length ? players : currentFallback(query, tour), summary: playerDatabaseSummary }, {
      headers: { "Cache-Control": "public, max-age=60, s-maxage=300, stale-while-revalidate=86400" },
    });
  } catch {
    return NextResponse.json({ players: currentFallback(query, tour), summary: playerDatabaseSummary, fallback: true }, {
      headers: { "Cache-Control": "public, max-age=30" },
    });
  }
}

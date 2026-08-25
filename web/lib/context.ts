import type { PlayerProfile, Surface } from "./model";

export type ContextConfidence = "verified" | "corroborated" | "reported" | "estimated";
export type NewsSignalKind = "availability" | "injury" | "coaching";

export interface VenueConditions {
  location: string;
  latitude: number;
  longitude: number;
  elevationM: number;
  timezone: string;
  observedAt: string;
  temperatureF: number;
  apparentTemperatureF: number;
  humidityPercent: number;
  precipitationIn: number;
  windMph: number;
  gustMph: number;
  weatherCode: number;
  sourceUrl: string;
}

export interface TravelEstimate {
  previousEvent: string | null;
  previousVenue: string | null;
  lastPlayedAt: string | null;
  daysRest: number | null;
  distanceKm: number | null;
  estimatedTimezoneShift: number | null;
  confidence: ContextConfidence;
}

export interface NewsSignal {
  id: string;
  player: string;
  kind: NewsSignalKind;
  severity: "watch" | "material" | "high";
  direction: "adverse" | "neutral";
  title: string;
  source: string;
  url: string;
  publishedAt: string;
  confidence: ContextConfidence;
}

export interface ContextFactor {
  id: string;
  label: string;
  detail: string;
  ratingPoints: number;
  uncertaintyPoints: number;
  servePointShift: number;
  confidence: ContextConfidence;
}

export interface PlayerContextAdjustment {
  player: string;
  availability: "available" | "questionable" | "withdrawn";
  ratingDelta: number;
  uncertaintyDelta: number;
  servePointShift: number;
  travel: TravelEstimate;
  news: NewsSignal[];
  factors: ContextFactor[];
}

export interface ContextReport {
  generatedAt: string;
  eventName: string;
  venue: string;
  surface: Surface;
  conditions: VenueConditions | null;
  players: [PlayerContextAdjustment, PlayerContextAdjustment];
  limitations: string[];
}

const clamp = (value: number, low: number, high: number) => Math.max(low, Math.min(high, value));

function emptyTravel(): TravelEstimate {
  return {
    previousEvent: null,
    previousVenue: null,
    lastPlayedAt: null,
    daysRest: null,
    distanceKm: null,
    estimatedTimezoneShift: null,
    confidence: "estimated",
  };
}

function travelFactors(travel: TravelEstimate): ContextFactor[] {
  const factors: ContextFactor[] = [];
  const rest = travel.daysRest;
  const distance = travel.distanceKm;
  const timezones = travel.estimatedTimezoneShift;

  if (rest !== null) {
    const ratingPoints = rest < 1.25 ? -24 : rest < 2.5 ? -12 : rest >= 4 && rest <= 8 ? 4 : 0;
    factors.push({
      id: "turnaround",
      label: "Recovery window",
      detail: `${rest.toFixed(1)} days since the latest indexed match${travel.previousEvent ? ` at ${travel.previousEvent}` : ""}.`,
      ratingPoints,
      uncertaintyPoints: rest < 2.5 ? 8 : 0,
      servePointShift: 0,
      confidence: travel.confidence,
    });
  }

  if (distance !== null && rest !== null && distance > 1_500) {
    const ratingPoints = distance > 8_000 && rest < 4 ? -18 : distance > 4_000 && rest < 3 ? -12 : rest < 2 ? -7 : 0;
    factors.push({
      id: "travel",
      label: "Travel load",
      detail: `Approximately ${Math.round(distance).toLocaleString()} km from the previous indexed venue${timezones !== null ? ` and about ${Math.abs(timezones)} time zones` : ""}.`,
      ratingPoints,
      uncertaintyPoints: ratingPoints < 0 ? 9 : 3,
      servePointShift: 0,
      confidence: "estimated",
    });
  }
  return factors;
}

function newsFactors(signals: NewsSignal[]): ContextFactor[] {
  const factors: ContextFactor[] = [];
  const adverse = signals.filter((signal) => signal.direction === "adverse" && (signal.kind === "injury" || signal.kind === "availability"));
  const coaching = signals.filter((signal) => signal.kind === "coaching");
  const recovery = signals.filter((signal) => signal.direction === "neutral" && (signal.kind === "injury" || signal.kind === "availability"));
  if (adverse.length) {
    const trusted = adverse.filter((signal) => signal.confidence === "verified");
    const domains = new Set(adverse.map((signal) => signal.source));
    const confidence: ContextConfidence = trusted.length ? "verified" : domains.size >= 2 ? "corroborated" : "reported";
    const high = adverse.some((signal) => signal.severity === "high");
    const material = adverse.some((signal) => signal.severity === "material");
    const ratingPoints = confidence === "reported" ? 0 : high ? -44 : material ? -22 : -8;
    factors.push({
      id: "availability-news",
      label: "Availability reporting",
      detail: `${adverse.length} recent relevant headline${adverse.length === 1 ? "" : "s"}; directional adjustment requires a trusted source or independent corroboration.`,
      ratingPoints,
      uncertaintyPoints: high ? 34 : material ? 22 : 12,
      servePointShift: 0,
      confidence,
    });
  }
  if (coaching.length) {
    factors.push({
      id: "coaching-news",
      label: "Coaching change",
      detail: `${coaching.length} recent coaching-change signal${coaching.length === 1 ? "" : "s"}; the model widens uncertainty without guessing whether the change is positive or negative.`,
      ratingPoints: 0,
      uncertaintyPoints: 16,
      servePointShift: 0,
      confidence: coaching.some((signal) => signal.confidence === "verified") ? "verified" : "reported",
    });
  }
  if (recovery.length) {
    factors.push({
      id: "recovery-news",
      label: "Return-to-play reporting",
      detail: `${recovery.length} recent recovery or return signal${recovery.length === 1 ? "" : "s"}; direction stays neutral while uncertainty reflects incomplete medical detail.`,
      ratingPoints: 0,
      uncertaintyPoints: 8,
      servePointShift: 0,
      confidence: recovery.some((signal) => signal.confidence === "verified") ? "verified" : "reported",
    });
  }
  return factors;
}

function conditionsFactor(profile: PlayerProfile, conditions: VenueConditions | null, indoor: boolean): ContextFactor[] {
  if (!conditions || indoor) return [];
  const factors: ContextFactor[] = [];
  const serverStyle = clamp((profile.servePointsWon - 0.59) / 0.09, -1, 1);
  if (conditions.elevationM >= 500) {
    const servePointShift = clamp(conditions.elevationM / 2_000 * 0.006 * (0.7 + serverStyle * 0.3), 0, 0.008);
    factors.push({
      id: "altitude",
      label: "Altitude",
      detail: `${Math.round(conditions.elevationM).toLocaleString()} m elevation is modeled as a modest serve-speed amplifier.`,
      ratingPoints: 0,
      uncertaintyPoints: 3,
      servePointShift,
      confidence: "estimated",
    });
  }
  if (conditions.windMph >= 12 || conditions.gustMph >= 20) {
    const servePointShift = -clamp((conditions.gustMph - 14) / 1_800 * (0.8 + Math.max(0, serverStyle) * 0.3), 0.001, 0.008);
    factors.push({
      id: "wind",
      label: "Wind exposure",
      detail: `${conditions.windMph.toFixed(0)} mph sustained wind with ${conditions.gustMph.toFixed(0)} mph gusts adds serve variance outdoors.`,
      ratingPoints: 0,
      uncertaintyPoints: 7,
      servePointShift,
      confidence: "estimated",
    });
  }
  if (conditions.apparentTemperatureF >= 90) {
    factors.push({
      id: "heat",
      label: "Heat stress",
      detail: `${conditions.apparentTemperatureF.toFixed(0)}°F apparent temperature increases physical uncertainty; no player-specific heat tolerance is assumed.`,
      ratingPoints: 0,
      uncertaintyPoints: 8,
      servePointShift: 0,
      confidence: "estimated",
    });
  }
  if (conditions.precipitationIn >= 0.02) {
    factors.push({
      id: "precipitation",
      label: "Weather disruption",
      detail: `${conditions.precipitationIn.toFixed(2)} in precipitation near the forecast hour can delay outdoor play or change court behavior.`,
      ratingPoints: 0,
      uncertaintyPoints: 10,
      servePointShift: 0,
      confidence: "estimated",
    });
  }
  return factors;
}

export function deriveContextAdjustment(options: {
  player: PlayerProfile;
  conditions: VenueConditions | null;
  travel?: TravelEstimate | null;
  news?: NewsSignal[];
  indoor?: boolean;
}): PlayerContextAdjustment {
  const travel = options.travel ?? emptyTravel();
  const news = options.news ?? [];
  const factors = [
    ...travelFactors(travel),
    ...newsFactors(news),
    ...conditionsFactor(options.player, options.conditions, Boolean(options.indoor)),
  ];
  const adverse = news.filter((signal) => signal.direction === "adverse" && (signal.kind === "injury" || signal.kind === "availability"));
  const trustedWithdrawal = adverse.some((signal) => signal.kind === "availability" && signal.severity === "high" && signal.confidence === "verified");
  const corroboratedWithdrawal = new Set(adverse.filter((signal) => signal.kind === "availability" && signal.severity === "high").map((signal) => signal.source)).size >= 2;
  return {
    player: options.player.name,
    availability: trustedWithdrawal || corroboratedWithdrawal ? "withdrawn" : adverse.length ? "questionable" : "available",
    ratingDelta: Math.round(clamp(factors.reduce((total, factor) => total + factor.ratingPoints, 0), -70, 18)),
    uncertaintyDelta: Math.round(clamp(factors.reduce((total, factor) => total + factor.uncertaintyPoints, 0), 0, 70)),
    servePointShift: clamp(factors.reduce((total, factor) => total + factor.servePointShift, 0), -0.012, 0.012),
    travel,
    news,
    factors,
  };
}

export function applyContextAdjustment(profile: PlayerProfile, adjustment: PlayerContextAdjustment): PlayerProfile {
  const rating = profile.rating + adjustment.ratingDelta;
  return {
    ...profile,
    rating,
    ratingSigma: clamp(profile.ratingSigma + adjustment.uncertaintyDelta, 18, 480),
    servePointsWon: clamp(profile.servePointsWon + adjustment.servePointShift, 0.40, 0.86),
    surfaceRating: {
      hard: profile.surfaceRating.hard + adjustment.ratingDelta,
      clay: profile.surfaceRating.clay + adjustment.ratingDelta,
      grass: profile.surfaceRating.grass + adjustment.ratingDelta,
    },
  };
}

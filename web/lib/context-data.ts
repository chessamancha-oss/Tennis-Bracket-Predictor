import {
  deriveContextAdjustment,
  type ContextReport,
  type NewsSignal,
  type PlayerContextAdjustment,
  type TravelEstimate,
  type VenueConditions,
} from "./context";
import type { PlayerProfile, Surface, Tour } from "./model";
import { normalizePlayerName } from "./player-database";

interface GeocodingResult {
  name?: string;
  country?: string;
  latitude?: number;
  longitude?: number;
  elevation?: number;
  timezone?: string;
}

interface GeocodingPayload {
  results?: GeocodingResult[];
}

interface ForecastPayload {
  elevation?: number;
  timezone?: string;
  utc_offset_seconds?: number;
  current?: WeatherPoint;
  hourly?: WeatherPoint & { time?: string[] };
}

interface NwsQuantity {
  unitCode?: string;
  value?: number | null;
}

interface NwsPointPayload {
  properties?: { observationStations?: string; timeZone?: string };
}

interface NwsStationPayload {
  features?: Array<{ properties?: { stationIdentifier?: string } }>;
}

interface NwsObservationPayload {
  properties?: {
    timestamp?: string;
    temperature?: NwsQuantity;
    heatIndex?: NwsQuantity;
    windChill?: NwsQuantity;
    relativeHumidity?: NwsQuantity;
    windSpeed?: NwsQuantity;
    windGust?: NwsQuantity;
    precipitationLastHour?: NwsQuantity | null;
    elevation?: NwsQuantity;
  };
}

interface WeatherPoint {
  time?: string | string[];
  temperature_2m?: number | number[];
  relative_humidity_2m?: number | number[];
  apparent_temperature?: number | number[];
  precipitation?: number | number[];
  weather_code?: number | number[];
  wind_speed_10m?: number | number[];
  wind_gusts_10m?: number | number[];
}

interface RawCompetitor {
  athlete?: { displayName?: string };
}

interface RawCompetition {
  date?: string;
  status?: { type?: { state?: string } };
  competitors?: RawCompetitor[];
}

interface RawGrouping {
  grouping?: { slug?: string };
  competitions?: RawCompetition[];
}

interface RawEvent {
  name?: string;
  date?: string;
  endDate?: string;
  venue?: { displayName?: string };
  groupings?: RawGrouping[];
}

interface ScoreboardPayload {
  events?: RawEvent[];
}

interface GdeltArticle {
  url?: string;
  title?: string;
  seendate?: string;
  domain?: string;
  language?: string;
}

const trustedNewsDomains = [
  "atptour.com",
  "wtatennis.com",
  "usopen.org",
  "ausopen.com",
  "rolandgarros.com",
  "wimbledon.com",
  "reuters.com",
  "apnews.com",
  "bbc.com",
  "bbc.co.uk",
  "espn.com",
  "skysports.com",
  "theguardian.com",
  "cbssports.com",
  "si.com",
  "tennis.com",
];

const knownVenueLocations: Record<string, GeocodingResult> = {
  "new york": { name: "New York", country: "United States", latitude: 40.7128, longitude: -74.006, timezone: "America/New_York" },
  "winston salem": { name: "Winston-Salem", country: "United States", latitude: 36.0999, longitude: -80.2442, timezone: "America/New_York" },
  philadelphia: { name: "Philadelphia", country: "United States", latitude: 39.9526, longitude: -75.1652, timezone: "America/New_York" },
  cincinnati: { name: "Cincinnati", country: "United States", latitude: 39.1031, longitude: -84.512, timezone: "America/New_York" },
  monterrey: { name: "Monterrey", country: "Mexico", latitude: 25.6866, longitude: -100.3161, timezone: "America/Monterrey" },
  montreal: { name: "Montreal", country: "Canada", latitude: 45.5017, longitude: -73.5673, timezone: "America/Toronto" },
  toronto: { name: "Toronto", country: "Canada", latitude: 43.6532, longitude: -79.3832, timezone: "America/Toronto" },
  "indian wells": { name: "Indian Wells", country: "United States", latitude: 33.7176, longitude: -116.3408, timezone: "America/Los_Angeles" },
  miami: { name: "Miami", country: "United States", latitude: 25.7617, longitude: -80.1918, timezone: "America/New_York" },
  madrid: { name: "Madrid", country: "Spain", latitude: 40.4168, longitude: -3.7038, timezone: "Europe/Madrid" },
  rome: { name: "Rome", country: "Italy", latitude: 41.9028, longitude: 12.4964, timezone: "Europe/Rome" },
  paris: { name: "Paris", country: "France", latitude: 48.8566, longitude: 2.3522, timezone: "Europe/Paris" },
  london: { name: "London", country: "United Kingdom", latitude: 51.5074, longitude: -0.1278, timezone: "Europe/London" },
  melbourne: { name: "Melbourne", country: "Australia", latitude: -37.8136, longitude: 144.9631, timezone: "Australia/Melbourne" },
};

function venueKey(value: string) {
  return value.toLowerCase().replace(/[^a-z0-9]+/g, " ").trim();
}

function isoDay(date: Date) {
  return date.toISOString().slice(0, 10).replaceAll("-", "");
}

function subtractDays(date: Date, days: number) {
  return new Date(date.getTime() - days * 86_400_000);
}

async function getJson<T>(url: string): Promise<T> {
  const response = await fetch(url, {
    headers: { Accept: "application/json", "User-Agent": "curl/8.7.1" },
    cache: "no-store",
  });
  if (!response.ok) throw new Error(`Source returned ${response.status}`);
  return response.json() as Promise<T>;
}

async function getText(url: string) {
  const response = await fetch(url, {
    headers: { Accept: "application/rss+xml, application/xml, text/xml", "User-Agent": "curl/8.7.1" },
    cache: "no-store",
  });
  if (!response.ok) throw new Error(`Source returned ${response.status}`);
  return response.text();
}

async function getNwsJson<T>(url: string): Promise<T> {
  const response = await fetch(url, {
    headers: {
      Accept: "application/geo+json",
      "User-Agent": "BaselineTennisLabs/3.1 (github.com/chessamancha-oss/Tennis-Bracket-Predictor)",
    },
    cache: "no-store",
  });
  if (!response.ok) throw new Error(`National Weather Service returned ${response.status}`);
  return response.json() as Promise<T>;
}

async function geocode(location: string) {
  const candidates = [location, location.split(",")[0]].map((item) => item.trim()).filter(Boolean);
  for (const candidate of candidates) {
    const known = knownVenueLocations[venueKey(candidate)];
    if (known) return known;
  }
  for (const candidate of candidates) {
    const params = new URLSearchParams({ name: candidate, count: "1", language: "en", format: "json" });
    try {
      const payload = await getJson<GeocodingPayload>(`https://geocoding-api.open-meteo.com/v1/search?${params}`);
      const result = payload.results?.[0];
      if (result?.latitude !== undefined && result.longitude !== undefined) return result;
    } catch {
      // Try the less-specific venue candidate before returning no coordinates.
    }
  }
  return null;
}

function numericAt(value: number | number[] | undefined, index: number, fallback = 0) {
  if (Array.isArray(value)) return Number(value[index] ?? fallback);
  return Number(value ?? fallback);
}

function weatherPoint(payload: ForecastPayload, startsAt?: string | null) {
  const hourlyTimes = payload.hourly?.time;
  if (startsAt && Array.isArray(hourlyTimes) && hourlyTimes.length) {
    const offset = payload.utc_offset_seconds ?? 0;
    const localTarget = new Date(new Date(startsAt).getTime() + offset * 1_000).toISOString().slice(0, 13);
    const index = hourlyTimes.findIndex((time) => time.startsWith(localTarget));
    if (index >= 0) return { source: payload.hourly ?? {}, index, observedAt: hourlyTimes[index] };
  }
  const currentTime = typeof payload.current?.time === "string" ? payload.current.time : new Date().toISOString();
  return { source: payload.current ?? {}, index: 0, observedAt: currentTime };
}

function quantityValue(quantity?: NwsQuantity | null) {
  return typeof quantity?.value === "number" && Number.isFinite(quantity.value) ? quantity.value : null;
}

function temperatureF(quantity?: NwsQuantity | null) {
  const value = quantityValue(quantity);
  if (value === null) return null;
  if (quantity?.unitCode?.includes("degC")) return value * 9 / 5 + 32;
  return value;
}

function windMph(quantity?: NwsQuantity | null) {
  const value = quantityValue(quantity);
  if (value === null) return null;
  if (quantity?.unitCode?.includes("km_h")) return value * 0.621371;
  if (quantity?.unitCode?.includes("m_s")) return value * 2.23694;
  return value;
}

function precipitationIn(quantity?: NwsQuantity | null) {
  const value = quantityValue(quantity);
  if (value === null) return 0;
  if (quantity?.unitCode?.endsWith(":m")) return value * 39.3701;
  if (quantity?.unitCode?.includes("mm")) return value * 0.0393701;
  return value;
}

async function conditionsFromNws(location: GeocodingResult, venue: string): Promise<VenueConditions | null> {
  if (location.latitude === undefined || location.longitude === undefined) return null;
  try {
    const point = await getNwsJson<NwsPointPayload>(`https://api.weather.gov/points/${location.latitude},${location.longitude}`);
    const stationsUrl = point.properties?.observationStations;
    if (!stationsUrl?.startsWith("https://api.weather.gov/")) return null;
    const stations = await getNwsJson<NwsStationPayload>(stationsUrl);
    const station = stations.features?.[0]?.properties?.stationIdentifier;
    if (!station) return null;
    const sourceUrl = `https://api.weather.gov/stations/${encodeURIComponent(station)}/observations/latest?require_qc=true`;
    const observation = await getNwsJson<NwsObservationPayload>(sourceUrl);
    const properties = observation.properties;
    const temperature = temperatureF(properties?.temperature);
    if (!properties || temperature === null) return null;
    const apparent = temperatureF(properties.heatIndex) ?? temperatureF(properties.windChill) ?? temperature;
    const wind = windMph(properties.windSpeed) ?? 0;
    const gust = windMph(properties.windGust) ?? wind;
    const precipitation = precipitationIn(properties.precipitationLastHour);
    return {
      location: [location.name, location.country].filter(Boolean).join(", ") || venue,
      latitude: location.latitude,
      longitude: location.longitude,
      elevationM: quantityValue(properties.elevation) ?? Number(location.elevation ?? 0),
      timezone: point.properties?.timeZone ?? location.timezone ?? "UTC",
      observedAt: properties.timestamp ?? new Date().toISOString(),
      temperatureF: temperature,
      apparentTemperatureF: apparent,
      humidityPercent: quantityValue(properties.relativeHumidity) ?? 0,
      precipitationIn: precipitation,
      windMph: wind,
      gustMph: gust,
      weatherCode: precipitation > 0 ? 61 : 0,
      sourceUrl,
    };
  } catch {
    return null;
  }
}

export async function conditionsAtVenue(venue: string, startsAt?: string | null): Promise<VenueConditions | null> {
  const location = await geocode(venue);
  if (location?.latitude === undefined || location.longitude === undefined) return null;
  if (["united states", "united states of america", "usa"].includes(venueKey(location.country ?? ""))) {
    const nationalWeatherService = await conditionsFromNws(location, venue);
    if (nationalWeatherService) return nationalWeatherService;
  }
  const shared = {
    latitude: String(location.latitude),
    longitude: String(location.longitude),
    current: "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m,wind_gusts_10m",
    temperature_unit: "fahrenheit",
    wind_speed_unit: "mph",
    precipitation_unit: "inch",
    timezone: "auto",
  };
  const detailed = new URLSearchParams({
    ...shared,
    hourly: "temperature_2m,relative_humidity_2m,apparent_temperature,precipitation,weather_code,wind_speed_10m,wind_gusts_10m",
    forecast_days: "16",
    past_days: "1",
  });
  const currentOnly = new URLSearchParams(shared);
  for (const params of [detailed, currentOnly]) {
    try {
      const payload = await getJson<ForecastPayload>(`https://api.open-meteo.com/v1/forecast?${params}`);
      const point = weatherPoint(payload, startsAt);
      return {
        location: [location.name, location.country].filter(Boolean).join(", ") || venue,
        latitude: location.latitude,
        longitude: location.longitude,
        elevationM: Number(payload.elevation ?? location.elevation ?? 0),
        timezone: payload.timezone ?? location.timezone ?? "UTC",
        observedAt: point.observedAt,
        temperatureF: numericAt(point.source.temperature_2m, point.index),
        apparentTemperatureF: numericAt(point.source.apparent_temperature, point.index),
        humidityPercent: numericAt(point.source.relative_humidity_2m, point.index),
        precipitationIn: numericAt(point.source.precipitation, point.index),
        windMph: numericAt(point.source.wind_speed_10m, point.index),
        gustMph: numericAt(point.source.wind_gusts_10m, point.index),
        weatherCode: numericAt(point.source.weather_code, point.index),
        sourceUrl: "https://open-meteo.com/en/docs",
      };
    } catch {
      // A smaller current-only request is a resilient fallback for edge runtimes.
    }
  }
  return null;
}

function haversine(first: GeocodingResult, second: GeocodingResult) {
  if (first.latitude === undefined || first.longitude === undefined || second.latitude === undefined || second.longitude === undefined) return null;
  const radians = (value: number) => value * Math.PI / 180;
  const deltaLat = radians(second.latitude - first.latitude);
  const deltaLon = radians(second.longitude - first.longitude);
  const startLat = radians(first.latitude);
  const endLat = radians(second.latitude);
  const value = Math.sin(deltaLat / 2) ** 2 + Math.cos(startLat) * Math.cos(endLat) * Math.sin(deltaLon / 2) ** 2;
  return 6_371 * 2 * Math.atan2(Math.sqrt(value), Math.sqrt(1 - value));
}

async function recentTravel(players: PlayerProfile[], tour: Tour, currentVenue: string, eventName: string, startsAt?: string | null) {
  const target = startsAt ? new Date(startsAt) : new Date();
  const range = `${isoDay(subtractDays(target, 21))}-${isoDay(target)}`;
  const desiredGrouping = tour === "ATP" ? "mens-singles" : "womens-singles";
  const output = new Map<string, TravelEstimate>();
  const empty = (): TravelEstimate => ({ previousEvent: null, previousVenue: null, lastPlayedAt: null, daysRest: null, distanceKm: null, estimatedTimezoneShift: null, confidence: "estimated" });
  players.forEach((player) => output.set(normalizePlayerName(player.name), empty()));
  try {
    const board = await getJson<ScoreboardPayload>(`https://site.api.espn.com/apis/site/v2/sports/tennis/${tour.toLowerCase()}/scoreboard?dates=${range}`);
    for (const event of board.events ?? []) {
      if (normalizePlayerName(event.name ?? "") === normalizePlayerName(eventName)) continue;
      const grouping = event.groupings?.find((item) => item.grouping?.slug === desiredGrouping);
      for (const match of grouping?.competitions ?? []) {
        if (match.status?.type?.state !== "post" || !match.date || new Date(match.date) >= target) continue;
        for (const competitor of match.competitors ?? []) {
          const key = normalizePlayerName(competitor.athlete?.displayName ?? "");
          const existing = output.get(key);
          if (!existing || existing.lastPlayedAt && new Date(existing.lastPlayedAt) >= new Date(match.date)) continue;
          output.set(key, {
            previousEvent: event.name ?? "Previous tour event",
            previousVenue: event.venue?.displayName ?? null,
            lastPlayedAt: match.date,
            daysRest: Math.max(0, (target.getTime() - new Date(match.date).getTime()) / 86_400_000),
            distanceKm: null,
            estimatedTimezoneShift: null,
            confidence: "estimated",
          });
        }
      }
    }
    const currentLocation = await geocode(currentVenue);
    await Promise.all([...output.entries()].map(async ([key, estimate]) => {
      if (!currentLocation || !estimate.previousVenue) return;
      const previous = await geocode(estimate.previousVenue);
      if (!previous) return;
      estimate.distanceKm = haversine(previous, currentLocation);
      if (previous.longitude !== undefined && currentLocation.longitude !== undefined) {
        estimate.estimatedTimezoneShift = Math.min(12, Math.round(Math.abs(currentLocation.longitude - previous.longitude) / 15));
      }
      output.set(key, estimate);
    }));
  } catch {
    // Missing travel history is represented explicitly rather than invented.
  }
  return output;
}

function parseSeenDate(value: string | undefined) {
  if (!value) return new Date().toISOString();
  if (/^\d{8}T\d{6}Z$/.test(value)) return `${value.slice(0, 4)}-${value.slice(4, 6)}-${value.slice(6, 8)}T${value.slice(9, 11)}:${value.slice(11, 13)}:${value.slice(13, 15)}Z`;
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? new Date().toISOString() : parsed.toISOString();
}

function decodeXml(value: string) {
  return value
    .replaceAll("<![CDATA[", "")
    .replaceAll("]]>", "")
    .replaceAll("&quot;", "\"")
    .replaceAll("&apos;", "'")
    .replaceAll("&lt;", "<")
    .replaceAll("&gt;", ">")
    .replaceAll("&amp;", "&");
}

function xmlTag(item: string, tag: string) {
  return decodeXml(item.match(new RegExp(`<${tag}(?:\\s[^>]*)?>([\\s\\S]*?)</${tag}>`, "i"))?.[1]?.trim() ?? "");
}

function parseNewsRss(xml: string): GdeltArticle[] {
  return [...xml.matchAll(/<item>([\s\S]*?)<\/item>/gi)].map((match) => {
    const item = match[1];
    const sourceUrl = decodeXml(item.match(/<source[^>]*url="([^"]+)"/i)?.[1] ?? "");
    let domain = xmlTag(item, "source");
    try {
      if (sourceUrl) domain = new URL(sourceUrl).hostname.replace(/^www\./, "");
    } catch {
      // The visible source name remains useful if its URL is malformed.
    }
    return {
      title: xmlTag(item, "title"),
      url: xmlTag(item, "link"),
      seendate: xmlTag(item, "pubDate"),
      domain,
      language: "English",
    };
  });
}

function classifyArticle(article: GdeltArticle, players: PlayerProfile[], eventName: string): NewsSignal | null {
  const title = article.title?.trim();
  const url = article.url?.trim();
  if (!title || !url) return null;
  const normalizedTitle = normalizePlayerName(title);
  const player = players.find((item) => {
    const full = normalizePlayerName(item.name);
    const surname = full.split(" ").at(-1) ?? full;
    return normalizedTitle.includes(full) || surname.length >= 4 && normalizedTitle.includes(surname);
  });
  if (!player) return null;
  const lower = title.toLowerCase();
  const recovery = /return(?:s|ed|ing)? (?:from|after).*?(?:injur|illness)|comeback from .*injur|injury comeback|after (?:an? )?injury|injury layoff|gears up.*injur|cleared|declared fit|ready to play|recovered from/.test(lower);
  const speculative = /tipped|could|may |might|doubt|uncertain|rumou?r|question mark|expected to|risk of injur|injury risk|believes|claims|would have|warns?|opinion/.test(lower);
  const high = /withdraw(?:s|al|n)?|withdrew|ruled out|will miss|out of .*open|surgery|hospital/.test(lower);
  const injury = /injur|illness|sick|strain|pain|medical|fitness doubt|health/.test(lower);
  const coaching = /(coach|coaching).*(split|change|appoint|hire|part|leave|exit)|(?:split|part ways|appoint|hire).*(coach|coaching)/.test(lower);
  if (!high && !injury && !coaching && !recovery) return null;
  const namedEvents = ["us open", "australian open", "french open", "roland garros", "wimbledon", "cincinnati", "winston salem", "indian wells", "miami open", "madrid open", "italian open"];
  const currentEvent = normalizePlayerName(eventName);
  const mentionsCurrentEvent = normalizedTitle.includes(currentEvent);
  const mentionsDifferentEvent = namedEvents.some((name) => normalizedTitle.includes(name) && !currentEvent.includes(name));
  const domain = (article.domain ?? "news source").replace(/^www\./, "");
  const confidence = trustedNewsDomains.some((trusted) => domain === trusted || domain.endsWith(`.${trusted}`)) ? "verified" : "reported";
  const kind = coaching ? "coaching" : high ? "availability" : "injury";
  return {
    id: `${normalizePlayerName(player.name)}-${normalizePlayerName(title).slice(0, 48)}`,
    player: player.name,
    kind,
    severity: high ? "high" : injury ? "material" : "watch",
    direction: recovery || speculative || coaching || mentionsDifferentEvent && !mentionsCurrentEvent ? "neutral" : "adverse",
    title,
    source: domain,
    url,
    publishedAt: parseSeenDate(article.seendate),
    confidence,
  };
}

async function relevantNews(players: PlayerProfile[], eventName: string) {
  const names = players.map((player) => `"${player.name.replaceAll("\"", "")}"`).join(" OR ");
  const search = async (query: string) => {
    const params = new URLSearchParams({ q: query, hl: "en-US", gl: "US", ceid: "US:en" });
    return parseNewsRss(await getText(`https://news.google.com/rss/search?${params}`));
  };
  try {
    const [statusArticles, coachingArticles] = await Promise.all([
      search(`(${names}) tennis (injury OR injured OR illness OR withdrawal OR withdrew OR medical OR fitness) when:14d`),
      search(`(${names}) tennis (coach OR coaching) (split OR change OR hire OR appoint OR "part ways") when:45d`),
    ]);
    const statusSignals = statusArticles.map((article) => classifyArticle(article, players, eventName)).filter((signal): signal is NewsSignal => signal !== null);
    const coachingSignals = coachingArticles.map((article) => classifyArticle(article, players, eventName)).filter((signal): signal is NewsSignal => signal?.kind === "coaching");
    const reliableCoachingSignals = coachingSignals.filter((signal) => signal.confidence === "verified" || new Set(coachingSignals.filter((item) => normalizePlayerName(item.player) === normalizePlayerName(signal.player)).map((item) => item.source)).size >= 2);
    const signals = [...statusSignals, ...reliableCoachingSignals];
    const unique = new Map(signals.map((signal) => [`${signal.player}|${signal.title}`, signal]));
    const priority = (signal: NewsSignal) => signal.direction === "adverse" ? 0 : signal.kind === "coaching" ? 1 : 2;
    return players.flatMap((player) => [...unique.values()]
      .filter((signal) => normalizePlayerName(signal.player) === normalizePlayerName(player.name))
      .sort((first, second) => priority(first) - priority(second) || Number(second.confidence === "verified") - Number(first.confidence === "verified") || second.publishedAt.localeCompare(first.publishedAt))
      .slice(0, 5));
  } catch {
    return [];
  }
}

function likelyIndoor(eventName: string) {
  const name = eventName.toLowerCase();
  return ["paris masters", "vienna", "basel", "stockholm", "rotterdam", "atp finals", "wta finals"].some((event) => name.includes(event));
}

export async function buildMatchContext(options: {
  players: [PlayerProfile, PlayerProfile];
  tour: Tour;
  surface: Surface;
  eventName: string;
  venue: string;
  startsAt?: string | null;
}): Promise<ContextReport> {
  const [conditions, travel, news] = await Promise.all([
    conditionsAtVenue(options.venue, options.startsAt),
    recentTravel(options.players, options.tour, options.venue, options.eventName, options.startsAt),
    relevantNews(options.players, options.eventName),
  ]);
  const indoor = likelyIndoor(options.eventName);
  const adjustments = options.players.map((player) => deriveContextAdjustment({
    player,
    conditions,
    travel: travel.get(normalizePlayerName(player.name)),
    news: news.filter((signal) => normalizePlayerName(signal.player) === normalizePlayerName(player.name)),
    indoor,
  })) as [PlayerContextAdjustment, PlayerContextAdjustment];
  return {
    generatedAt: new Date().toISOString(),
    eventName: options.eventName,
    venue: options.venue,
    surface: options.surface,
    conditions,
    players: adjustments,
    limitations: [
      "News headlines are signals, not medical records; uncorroborated reports only widen uncertainty.",
      "Travel uses the latest completed match found in the preceding 21-day scoreboard window.",
      indoor ? "This event is treated as indoors, so outdoor weather is displayed but not applied." : "Weather is a venue-level forecast and may not represent the exact court or roof state.",
    ],
  };
}

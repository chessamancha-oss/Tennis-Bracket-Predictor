import type { Surface, Tour } from "./model";

const grassEvents = [
  "wimbledon",
  "queen's",
  "queens",
  "halle",
  "eastbourne",
  "nottingham",
  "bad homburg",
  "mallorca",
  "s-hertogenbosch",
];

const clayEvents = [
  "roland garros",
  "french open",
  "monte-carlo",
  "monte carlo",
  "madrid",
  "rome",
  "internazionali bnl",
  "barcelona",
  "hamburg",
  "munich",
  "bastad",
  "gstaad",
  "umag",
  "kitzbuhel",
  "kitzbühel",
];

const majorEvents = ["australian open", "roland garros", "french open", "wimbledon", "us open", "u.s. open"];

export function inferTourSurface(eventName: string, venue: string, tour: Tour): Surface {
  const description = `${eventName} ${venue}`.toLowerCase();
  if (description.includes("stuttgart")) return tour === "ATP" ? "grass" : "clay";
  if (grassEvents.some((name) => description.includes(name))) return "grass";
  if (clayEvents.some((name) => description.includes(name))) return "clay";
  return "hard";
}

export function inferMatchFormat(eventName: string, round: string, tour: Tour): 3 | 5 {
  if (tour !== "ATP" || /qualif/i.test(round)) return 3;
  return majorEvents.some((name) => eventName.toLowerCase().includes(name)) ? 5 : 3;
}

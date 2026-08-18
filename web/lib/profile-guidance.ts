export type FactorKey = "serve" | "return" | "movement" | "clutch" | "form" | "fitness" | "surface" | "experience";

export type CustomRatings = Record<FactorKey, number>;

export const factorGuidance: Record<FactorKey, {
  label: string;
  short: string;
  low: string;
  middle: string;
  high: string;
  modelUse: string;
}> = {
  serve: {
    label: "Serve pressure",
    short: "Free points, first-strike control, and second-serve safety.",
    low: "1–3: serve is frequently attacked; breaks are expected.",
    middle: "4–7: competitive pace and placement, with ordinary pressure leaks.",
    high: "8–10: elite first-strike profile; many holds begin with an immediate edge.",
    modelUse: "Sets the center of a beta posterior for service points won, then allows uncertainty around it.",
  },
  return: {
    label: "Return quality",
    short: "Neutralizing first serves and taking control of second serves.",
    low: "1–3: struggles to start neutral rallies against quality serving.",
    middle: "4–7: gets a normal share of returns in play and pressures second serves.",
    high: "8–10: elite read and depth; even strong servers face repeated scoreboard pressure.",
    modelUse: "Shapes the opponent’s serve-point posterior rather than adding a flat bonus.",
  },
  movement: {
    label: "Movement & defense",
    short: "Court coverage, recovery, transition balance, and rally tolerance.",
    low: "1–3: short rallies are essential; wide movement exposes the court.",
    middle: "4–7: sound coverage with a few attackable movement patterns.",
    high: "8–10: turns apparent winners into neutral balls and sustains defense under load.",
    modelUse: "Improves the return distribution and resistance in longer, later-set games.",
  },
  clutch: {
    label: "Pressure execution",
    short: "Decision quality on deuce, break points, tiebreaks, and closing games.",
    low: "1–3: level tends to fall when points carry extra consequence.",
    middle: "4–7: pressure performance is close to baseline ability.",
    high: "8–10: unusually reliable patterns and shot tolerance in leverage points.",
    modelUse: "Changes only high-leverage point distributions—deuce, break-like points, and tiebreaks.",
  },
  form: {
    label: "Recent form",
    short: "The quality and consistency of the player’s latest competitive stretch.",
    low: "1–3: repeated early losses or visibly unstable execution.",
    middle: "4–7: mixed-to-positive results around the player’s normal level.",
    high: "8–10: deep runs and repeated wins over comparable or stronger opposition.",
    modelUse: "Moves the short-term latent rating while shrinking toward long-run ability.",
  },
  fitness: {
    label: "Fitness & durability",
    short: "Physical availability, repeat-sprint capacity, and late-match resilience.",
    low: "1–3: restricted workload or a meaningful chance of late-match fade.",
    middle: "4–7: match-ready with ordinary fatigue across a demanding contest.",
    high: "8–10: exceptional repeatability; level is likely to survive long sets and extra volume.",
    modelUse: "Controls set-by-set fatigue drift and widens uncertainty when availability is questionable.",
  },
  surface: {
    label: "Surface comfort",
    short: "Movement, timing, patterns, and results on the selected court.",
    low: "1–3: the bounce and movement disrupt preferred patterns.",
    middle: "4–7: competent and tactically adaptable on the surface.",
    high: "8–10: the surface amplifies the player’s best patterns and movement.",
    modelUse: "Creates a surface-specific latent rating that is partially pooled with overall ability.",
  },
  experience: {
    label: "Match experience",
    short: "Quality of competitive reps, tactical adaptation, and score management.",
    low: "1–3: limited reps against varied styles or unfamiliar match states.",
    middle: "4–7: dependable pattern recognition at a competitive level.",
    high: "8–10: extensive high-level reps and sophisticated in-match problem solving.",
    modelUse: "Tightens rating uncertainty and stabilizes the pressure-performance estimate.",
  },
};

const levelNames = ["", "raw", "limited", "developing", "serviceable", "balanced", "strong", "advanced", "excellent", "tour-class", "outlier"];

export function interpretation(key: FactorKey, score: number, playerName: string): string {
  const guidance = factorGuidance[key];
  const percentile = Math.min(99, Math.max(2, Math.round(5 + (score - 1) * 10.2)));
  const picture = score <= 3 ? guidance.low.split(": ")[1] : score <= 7 ? guidance.middle.split(": ")[1] : guidance.high.split(": ")[1];
  return `${playerName} is ${levelNames[score]} here (${score}/10, roughly the ${percentile}th percentile of this scale): ${picture} In the forecast, this ${guidance.modelUse.charAt(0).toLowerCase()}${guidance.modelUse.slice(1)}`;
}

export const defaultRatings: CustomRatings = {
  serve: 6,
  return: 6,
  movement: 6,
  clutch: 5,
  form: 6,
  fitness: 7,
  surface: 6,
  experience: 5,
};

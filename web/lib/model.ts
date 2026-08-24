import type { CustomRatings } from "./profile-guidance";

export type Surface = "hard" | "clay" | "grass";
export type Tour = "ATP" | "WTA";

export interface PlayerProfile {
  id: string;
  tour: Tour | "CUSTOM";
  name: string;
  rank: number | null;
  rankingPoints: number | null;
  country: string;
  hand: string;
  age: number | null;
  rating: number;
  surfaceRating: Record<Surface, number>;
  ratingSigma: number;
  matches52w: number;
  wins52w: number;
  form90d: number;
  servePointsWon: number;
  returnPointsWon: number;
  holdRate: number;
  aceRate: number | null;
  doubleFaultRate: number | null;
  serveSample: number;
  returnSample: number;
  surfaceSamples: Record<Surface, number>;
  lastMatchDate: string | null;
  rankingSnapshot: string | null;
  historyCutoff: string | null;
  custom?: CustomRatings;
}

export interface EvidenceItem {
  label: string;
  leader: "one" | "two" | "even";
  detail: string;
  strength: number;
}

export interface PredictionResult {
  playerOneProbability: number;
  playerTwoProbability: number;
  intervalLow: number;
  intervalHigh: number;
  projectedWinner: string;
  likelySetScore: string;
  expectedSets: number;
  expectedGames: number;
  tieBreakChance: number;
  averageServePointOne: number;
  averageServePointTwo: number;
  simulations: number;
  posteriorDraws: number;
  confidence: "Exploratory" | "Moderate" | "Strong";
  evidence: EvidenceItem[];
}

export interface AdvancedProfileInputs {
  rating: number;
  surfaceRating: number;
  ratingUncertainty: number;
  servePointsWon: number;
  returnPointsWon: number;
  formRate: number;
  sampleMatches: number;
  clutchIndex: number;
  fitnessIndex: number;
}

interface DrawnSkills {
  serveOne: number;
  serveTwo: number;
  clutchOne: number;
  clutchTwo: number;
  fitnessOne: number;
  fitnessTwo: number;
}

interface MatchOutcome {
  winner: 1 | 2;
  setsOne: number;
  setsTwo: number;
  games: number;
  tieBreak: boolean;
}

const clamp = (value: number, low: number, high: number) => Math.max(low, Math.min(high, value));
function seedFrom(text: string): number {
  let value = 2166136261;
  for (let index = 0; index < text.length; index += 1) {
    value ^= text.charCodeAt(index);
    value = Math.imul(value, 16777619);
  }
  return value >>> 0;
}

function makeRandom(seed: number) {
  let state = seed || 1;
  return () => {
    state += 0x6d2b79f5;
    let value = state;
    value = Math.imul(value ^ (value >>> 15), value | 1);
    value ^= value + Math.imul(value ^ (value >>> 7), value | 61);
    return ((value ^ (value >>> 14)) >>> 0) / 4294967296;
  };
}

function normal(rng: () => number): number {
  const first = Math.max(rng(), Number.EPSILON);
  return Math.sqrt(-2 * Math.log(first)) * Math.cos(2 * Math.PI * rng());
}

function gamma(shape: number, rng: () => number): number {
  if (shape < 1) return gamma(shape + 1, rng) * Math.pow(rng(), 1 / shape);
  const d = shape - 1 / 3;
  const c = 1 / Math.sqrt(9 * d);
  for (;;) {
    let x: number;
    let v: number;
    do { x = normal(rng); v = Math.pow(1 + c * x, 3); } while (v <= 0);
    const u = rng();
    if (u < 1 - 0.0331 * x ** 4 || Math.log(u) < 0.5 * x * x + d * (1 - v + Math.log(v))) return d * v;
  }
}

function beta(mean: number, observations: number, priorMean: number, rng: () => number): number {
  const effective = Math.min(650, Math.max(20, observations));
  const priorStrength = 90;
  const alpha = clamp(mean, 0.05, 0.95) * effective + priorMean * priorStrength;
  const betaShape = (1 - clamp(mean, 0.05, 0.95)) * effective + (1 - priorMean) * priorStrength;
  const first = gamma(alpha, rng);
  return first / (first + gamma(betaShape, rng));
}

function quantile(values: number[], probability: number): number {
  const sorted = [...values].sort((a, b) => a - b);
  const index = (sorted.length - 1) * probability;
  const lower = Math.floor(index);
  const remainder = index - lower;
  return sorted[lower] + (sorted[lower + 1] !== undefined ? remainder * (sorted[lower + 1] - sorted[lower]) : 0);
}

function highLeverageShift(serverClutch: number, returnClutch: number) {
  return (serverClutch - returnClutch) * 0.0055;
}

function simulateGame(pointProbability: number, serverClutch: number, returnClutch: number, rng: () => number): boolean {
  let server = 0;
  let receiver = 0;
  for (let point = 0; point < 48; point += 1) {
    const leverage = server >= 3 || receiver >= 3;
    const adjusted = clamp(pointProbability + (leverage ? highLeverageShift(serverClutch, returnClutch) : 0), 0.38, 0.86);
    if (rng() < adjusted) server += 1; else receiver += 1;
    if ((server >= 4 || receiver >= 4) && Math.abs(server - receiver) >= 2) return server > receiver;
  }
  return rng() < pointProbability;
}

function simulateTieBreak(skills: DrawnSkills, firstServer: 1 | 2, rng: () => number): 1 | 2 {
  let one = 0;
  let two = 0;
  for (let point = 0; point < 60; point += 1) {
    const block = point === 0 ? 0 : Math.floor((point + 1) / 2);
    const server = (block % 2 === 0 ? firstServer : firstServer === 1 ? 2 : 1) as 1 | 2;
    const pOne = server === 1
      ? clamp(skills.serveOne + highLeverageShift(skills.clutchOne, skills.clutchTwo), 0.38, 0.86)
      : 1 - clamp(skills.serveTwo + highLeverageShift(skills.clutchTwo, skills.clutchOne), 0.38, 0.86);
    if (rng() < pOne) one += 1; else two += 1;
    if ((one >= 7 || two >= 7) && Math.abs(one - two) >= 2) return one > two ? 1 : 2;
  }
  return rng() < 0.5 ? 1 : 2;
}

function simulateSet(skills: DrawnSkills, firstServer: 1 | 2, setIndex: number, rng: () => number) {
  let one = 0;
  let two = 0;
  let server = firstServer;
  const fatigueOne = (skills.fitnessOne - 5.5) * 0.0016 * setIndex;
  const fatigueTwo = (skills.fitnessTwo - 5.5) * 0.0016 * setIndex;
  while (one < 6 && two < 6 || Math.abs(one - two) < 2) {
    if (one === 6 && two === 6) {
      const winner = simulateTieBreak({ ...skills, serveOne: skills.serveOne + fatigueOne, serveTwo: skills.serveTwo + fatigueTwo }, server, rng);
      if (winner === 1) one += 1; else two += 1;
      return { winner, one, two, nextServer: server === 1 ? 2 as const : 1 as const, tieBreak: true };
    }
    const serverWon = server === 1
      ? simulateGame(skills.serveOne + fatigueOne, skills.clutchOne, skills.clutchTwo, rng)
      : simulateGame(skills.serveTwo + fatigueTwo, skills.clutchTwo, skills.clutchOne, rng);
    if ((server === 1 && serverWon) || (server === 2 && !serverWon)) one += 1; else two += 1;
    server = server === 1 ? 2 : 1;
    if ((one >= 6 || two >= 6) && Math.abs(one - two) >= 2) break;
  }
  return { winner: (one > two ? 1 : 2) as 1 | 2, one, two, nextServer: server, tieBreak: false };
}

function simulateMatch(skills: DrawnSkills, bestOf: 3 | 5, rng: () => number): MatchOutcome {
  const needed = Math.ceil(bestOf / 2);
  let setsOne = 0;
  let setsTwo = 0;
  let games = 0;
  let tieBreak = false;
  let firstServer = rng() < 0.5 ? 1 as const : 2 as const;
  let setIndex = 0;
  while (setsOne < needed && setsTwo < needed) {
    const result = simulateSet(skills, firstServer, setIndex, rng);
    setsOne += Number(result.winner === 1);
    setsTwo += Number(result.winner === 2);
    games += result.one + result.two;
    tieBreak ||= result.tieBreak;
    firstServer = result.nextServer;
    setIndex += 1;
  }
  return { winner: setsOne > setsTwo ? 1 : 2, setsOne, setsTwo, games, tieBreak };
}

function drawSkills(one: PlayerProfile, two: PlayerProfile, surface: Surface, rng: () => number): DrawnSkills {
  const baseServe = one.tour === "ATP" ? 0.64 : one.tour === "WTA" ? 0.59 : 0.61;
  const baseReturn = 1 - baseServe;
  const oneRating = one.surfaceRating[surface] + normal(rng) * one.ratingSigma;
  const twoRating = two.surfaceRating[surface] + normal(rng) * two.ratingSigma;
  const ratingPointShift = clamp((oneRating - twoRating) / 400 * 0.024, -0.045, 0.045);
  const oneServe = beta(one.servePointsWon, one.serveSample, baseServe, rng);
  const twoServe = beta(two.servePointsWon, two.serveSample, baseServe, rng);
  const oneReturn = beta(one.returnPointsWon, one.returnSample, baseReturn, rng);
  const twoReturn = beta(two.returnPointsWon, two.returnSample, baseReturn, rng);
  const movementOne = one.custom ? (one.custom.movement - 5.5) * 0.0035 : 0;
  const movementTwo = two.custom ? (two.custom.movement - 5.5) * 0.0035 : 0;
  return {
    serveOne: clamp(baseServe + (oneServe - baseServe) - (twoReturn + movementTwo - baseReturn) + ratingPointShift, 0.43, 0.82),
    serveTwo: clamp(baseServe + (twoServe - baseServe) - (oneReturn + movementOne - baseReturn) - ratingPointShift, 0.43, 0.82),
    clutchOne: one.custom?.clutch ?? 5.5,
    clutchTwo: two.custom?.clutch ?? 5.5,
    fitnessOne: one.custom?.fitness ?? 6.5,
    fitnessTwo: two.custom?.fitness ?? 6.5,
  };
}

function leader(delta: number): "one" | "two" | "even" {
  return Math.abs(delta) < 0.012 ? "even" : delta > 0 ? "one" : "two";
}

function evidence(one: PlayerProfile, two: PlayerProfile, surface: Surface): EvidenceItem[] {
  const ratingDelta = one.surfaceRating[surface] - two.surfaceRating[surface];
  const serveDelta = one.servePointsWon - two.servePointsWon;
  const returnDelta = one.returnPointsWon - two.returnPointsWon;
  const formDelta = one.form90d - two.form90d;
  return [
    { label: `${surface[0].toUpperCase()}${surface.slice(1)}-court rating`, leader: leader(ratingDelta / 400), detail: `${Math.abs(Math.round(ratingDelta))} rating-point separation after surface shrinkage`, strength: clamp(Math.abs(ratingDelta) / 180, 0.08, 1) },
    { label: "Serve-point posterior", leader: leader(serveDelta), detail: `${Math.abs(serveDelta * 100).toFixed(1)} percentage-point historical edge`, strength: clamp(Math.abs(serveDelta) / 0.08, 0.08, 1) },
    { label: "Return-point posterior", leader: leader(returnDelta), detail: `${Math.abs(returnDelta * 100).toFixed(1)} percentage-point historical edge`, strength: clamp(Math.abs(returnDelta) / 0.08, 0.08, 1) },
    { label: "90-day form", leader: leader(formDelta), detail: `${Math.abs(formDelta * 100).toFixed(0)} percentage-point result gap, partially pooled`, strength: clamp(Math.abs(formDelta) / 0.35, 0.08, 1) },
  ];
}

export function predictMatch(one: PlayerProfile, two: PlayerProfile, surface: Surface, bestOf: 3 | 5, requestedSimulations = 5040, requestedPosteriorDraws = 36): PredictionResult {
  const posteriorDraws = Math.max(6, Math.round(requestedPosteriorDraws));
  const simulationsPerDraw = Math.max(10, Math.round(requestedSimulations / posteriorDraws));
  const simulations = posteriorDraws * simulationsPerDraw;
  const rng = makeRandom(seedFrom(`${one.id}|${two.id}|${surface}|${bestOf}|v2`));
  const drawProbabilities: number[] = [];
  const scoreCounts = new Map<string, number>();
  let winsOne = 0;
  let games = 0;
  let sets = 0;
  let tieBreaks = 0;
  let serveOne = 0;
  let serveTwo = 0;

  for (let draw = 0; draw < posteriorDraws; draw += 1) {
    const skills = drawSkills(one, two, surface, rng);
    serveOne += skills.serveOne;
    serveTwo += skills.serveTwo;
    let drawWins = 0;
    for (let run = 0; run < simulationsPerDraw; run += 1) {
      const outcome = simulateMatch(skills, bestOf, rng);
      winsOne += Number(outcome.winner === 1);
      drawWins += Number(outcome.winner === 1);
      games += outcome.games;
      sets += outcome.setsOne + outcome.setsTwo;
      tieBreaks += Number(outcome.tieBreak);
      const score = outcome.winner === 1 ? `${outcome.setsOne}–${outcome.setsTwo}` : `${outcome.setsTwo}–${outcome.setsOne}`;
      scoreCounts.set(score, (scoreCounts.get(score) ?? 0) + 1);
    }
    drawProbabilities.push(drawWins / simulationsPerDraw);
  }

  const probability = winsOne / simulations;
  const likelySetScore = [...scoreCounts.entries()].sort((a, b) => b[1] - a[1])[0]?.[0] ?? "—";
  const intervalLow = quantile(drawProbabilities, 0.1);
  const intervalHigh = quantile(drawProbabilities, 0.9);
  const width = intervalHigh - intervalLow;
  const sampleFloor = Math.min(one.matches52w, two.matches52w);
  const confidence = width < 0.20 && sampleFloor >= 35 ? "Strong" : width < 0.34 && sampleFloor >= 16 ? "Moderate" : "Exploratory";

  return {
    playerOneProbability: probability,
    playerTwoProbability: 1 - probability,
    intervalLow,
    intervalHigh,
    projectedWinner: probability >= 0.5 ? one.name : two.name,
    likelySetScore,
    expectedSets: sets / simulations,
    expectedGames: games / simulations,
    tieBreakChance: tieBreaks / simulations,
    averageServePointOne: serveOne / posteriorDraws,
    averageServePointTwo: serveTwo / posteriorDraws,
    simulations,
    posteriorDraws,
    confidence,
    evidence: evidence(one, two, surface),
  };
}

function probabilityInput(value: number, fallback: number) {
  if (!Number.isFinite(value)) return fallback;
  return clamp(Math.abs(value) > 1 ? value / 100 : value, 0.01, 0.99);
}

function openIndex(value: number) {
  if (!Number.isFinite(value)) return 5.5;
  return 5.5 + 4.5 * Math.tanh(value / 3.5);
}

export function advancedProfile(id: string, name: string, inputs: AdvancedProfileInputs, surface: Surface): PlayerProfile {
  const rating = Number.isFinite(inputs.rating) ? inputs.rating : 1500;
  const selectedSurfaceRating = Number.isFinite(inputs.surfaceRating) ? inputs.surfaceRating : rating;
  const sample = Math.max(0, Number.isFinite(inputs.sampleMatches) ? inputs.sampleMatches : 0);
  const form = probabilityInput(inputs.formRate, 0.5);
  const serve = probabilityInput(inputs.servePointsWon, 0.61);
  const returns = probabilityInput(inputs.returnPointsWon, 0.39);
  const uncertainty = clamp(Math.abs(Number.isFinite(inputs.ratingUncertainty) ? inputs.ratingUncertainty : 140), 18, 420);
  const custom = {
    serve: 5.5,
    return: 5.5,
    movement: 5.5,
    clutch: openIndex(inputs.clutchIndex),
    form: 5.5,
    fitness: openIndex(inputs.fitnessIndex),
    surface: 5.5,
    experience: clamp(1 + Math.log10(sample + 1) * 3.2, 1, 10),
  };
  return {
    id,
    tour: "CUSTOM",
    name: name.trim() || (id.endsWith("one") ? "Player A" : "Player B"),
    rank: null,
    rankingPoints: null,
    country: "Custom model",
    hand: "Unspecified",
    age: null,
    rating,
    surfaceRating: {
      hard: surface === "hard" ? selectedSurfaceRating : rating,
      clay: surface === "clay" ? selectedSurfaceRating : rating,
      grass: surface === "grass" ? selectedSurfaceRating : rating,
    },
    ratingSigma: uncertainty,
    matches52w: Math.round(sample),
    wins52w: Math.round(sample * form),
    form90d: form,
    servePointsWon: serve,
    returnPointsWon: returns,
    holdRate: clamp(0.5 + (serve - 0.54) * 2.25, 0.35, 0.98),
    aceRate: null,
    doubleFaultRate: null,
    serveSample: Math.round(sample * 120),
    returnSample: Math.round(sample * 120),
    surfaceSamples: { hard: Math.round(sample / 2), clay: Math.round(sample / 3), grass: Math.round(sample / 6) },
    lastMatchDate: null,
    rankingSnapshot: null,
    historyCutoff: null,
    custom,
  };
}

export function customProfile(id: string, name: string, ratings: CustomRatings, surface: Surface): PlayerProfile {
  const average = Object.values(ratings).reduce((total, value) => total + value, 0) / Object.values(ratings).length;
  const rating = 1240 + average * 58 + (ratings.form - 5.5) * 12;
  const surfaceRating = rating + (ratings.surface - 5.5) * 20;
  return {
    id,
    tour: "CUSTOM",
    name: name.trim() || (id.endsWith("one") ? "Player A" : "Player B"),
    rank: null,
    rankingPoints: null,
    country: "Custom",
    hand: "Right",
    age: null,
    rating,
    surfaceRating: { hard: surface === "hard" ? surfaceRating : rating, clay: surface === "clay" ? surfaceRating : rating, grass: surface === "grass" ? surfaceRating : rating },
    ratingSigma: clamp(154 - ratings.experience * 5 - ratings.fitness * 1.5, 82, 145),
    matches52w: Math.round(5 + ratings.experience * 5),
    wins52w: Math.round((5 + ratings.experience * 5) * (0.24 + ratings.form * 0.054)),
    form90d: clamp(0.24 + ratings.form * 0.054, 0.25, 0.82),
    servePointsWon: clamp(0.49 + ratings.serve * 0.021, 0.51, 0.72),
    returnPointsWon: clamp(0.275 + ratings.return * 0.019, 0.29, 0.48),
    holdRate: clamp(0.43 + ratings.serve * 0.048, 0.48, 0.92),
    aceRate: null,
    doubleFaultRate: null,
    serveSample: 45 + ratings.experience * 18,
    returnSample: 45 + ratings.experience * 18,
    surfaceSamples: { hard: 4 + ratings.experience * 2, clay: 4 + ratings.experience * 2, grass: 4 + ratings.experience * 2 },
    lastMatchDate: null,
    rankingSnapshot: null,
    historyCutoff: null,
    custom: ratings,
  };
}

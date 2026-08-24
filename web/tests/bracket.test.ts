import assert from "node:assert/strict";
import test from "node:test";
import { forecastBracket } from "../lib/bracket";
import { advancedProfile } from "../lib/model";

const inputs = {
  rating: 1700,
  surfaceRating: 1725,
  ratingUncertainty: 110,
  servePointsWon: 63,
  returnPointsWon: 39,
  formRate: 60,
  sampleMatches: 30,
  clutchIndex: 0,
  fitnessIndex: 0,
};

test("advanced profiles accept open-ended values while producing stable probabilities", () => {
  const profile = advancedProfile("extreme-one", "Extreme", { ...inputs, rating: 4200, clutchIndex: 250, fitnessIndex: -80 }, "hard");
  assert.equal(profile.rating, 4200);
  assert.ok(profile.custom);
  assert.ok(Number.isFinite(profile.ratingSigma));
  assert.ok(profile.servePointsWon > 0 && profile.servePointsWon < 1);
});
test("non-power-of-two fields receive byes and resolve a complete bracket", () => {
  const participants = Array.from({ length: 6 }, (_, index) => {
    const name = `Player ${index + 1}`;
    return { id: `p-${index}`, name, profile: advancedProfile(`p-${index}`, name, { ...inputs, rating: 1800 - index * 35, surfaceRating: 1800 - index * 35 }, "hard") };
  });
  const bracket = forecastBracket(participants, "hard", 3);
  assert.equal(bracket.size, 8);
  assert.equal(bracket.rounds.length, 3);
  assert.equal(bracket.rounds[0].matches.filter((match) => match.bye).length, 2);
  assert.ok(bracket.champion);
  assert.equal(bracket.rounds.at(-1)?.label, "Final");
});

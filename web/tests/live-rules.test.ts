import assert from "node:assert/strict";
import test from "node:test";
import { inferMatchFormat, inferTourSurface } from "../lib/live-rules";

test("Stuttgart surface follows the tour-specific event", () => {
  assert.equal(inferTourSurface("BOSS Open Stuttgart", "Stuttgart", "ATP"), "grass");
  assert.equal(inferTourSurface("Porsche Tennis Grand Prix", "Stuttgart", "WTA"), "clay");
});

test("recognized tour stops resolve their court surface", () => {
  assert.equal(inferTourSurface("Wimbledon", "All England Club", "ATP"), "grass");
  assert.equal(inferTourSurface("Roland Garros", "Paris", "WTA"), "clay");
  assert.equal(inferTourSurface("US Open", "New York", "ATP"), "hard");
});

test("ATP major main draws use best of five while all other cases stay best of three", () => {
  assert.equal(inferMatchFormat("US Open", "Men's Singles - Round 1", "ATP"), 5);
  assert.equal(inferMatchFormat("Wimbledon", "Quarterfinal", "ATP"), 5);
  assert.equal(inferMatchFormat("Australian Open", "Qualifying 3rd Round", "ATP"), 3);
  assert.equal(inferMatchFormat("US Open", "Women's Singles - Final", "WTA"), 3);
  assert.equal(inferMatchFormat("Cincinnati Open", "Final", "ATP"), 3);
});

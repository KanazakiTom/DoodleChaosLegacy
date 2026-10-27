const fs = require('fs');
const path = require('path');
// This script expects two arguments: <ticks> <track_json_path>
// It loads the track, runs the Line Rider physics engine (lr-core) and prints
// an array of rider states for each tick (JSON on stdout).

if (process.argv.length < 4) {
  console.error('Usage: node run_sim.js <ticks> <track_json_path>');
  process.exit(1);
}

const ticks = parseInt(process.argv[2], 10);
const trackPath = process.argv[3];

// Load the track JSON
const trackData = JSON.parse(fs.readFileSync(trackPath, 'utf8'));

// -------------------------------------------------
// Initialize the physics engine from lr-core
// -------------------------------------------------
// We assume lr-core is installed as a sibling folder and can be required.
// The core exposes a `LineRiderEngine` class that accepts a track definition.
const { LineRiderEngine } = require('../lr-core/src/line-rider-engine'); // adjust path if needed

// Create engine instance
const engine = new LineRiderEngine({
  tickRate: 40, // 40 ticks per second as in the official web version
  gravity: 0.175,
});

// Load lines into the engine (engine expects an array of line objects with x1,y1,x2,y2,type)
engine.loadTrack(trackData.lines);

// Add riders – we only need the first rider for sync evaluation
engine.addRider({
  x: trackData.startPosition.x,
  y: trackData.startPosition.y,
  vx: trackData.riders?.[0]?.startVelocity?.x ?? 0.4,
  vy: trackData.riders?.[0]?.startVelocity?.y ?? 0,
});

// Simulate
let states = [];
for (let i = 0; i < ticks; i++) {
  engine.tick();
  const rider = engine.riders[0]; // assume single rider for now
  states.push({
    tick: i,
    x: rider.x,
    y: rider.y,
    vx: rider.vx,
    vy: rider.vy,
    offSled: rider.offSled || false,
  });
}

// Output as JSON to stdout
process.stdout.write(JSON.stringify(states, null, 2));

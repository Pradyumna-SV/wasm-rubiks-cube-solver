# wasm-rubiks-solver

A 2-phase Rubik's Cube solver written in Rust, compiled to WebAssembly. Solves arbitrary scrambles in under 40ms on average with solutions of 22 moves or fewer.

---

## How it works

Uses Herbert Kociemba's 2-phase algorithm with D4 symmetry group reductions to prune the search space. Redundant branches are eliminated via precomputed allowed-moves tables. The WASM build replaces unavailable system timing APIs with `Date.now()`.

**Two initialization modes:**

- **Embedded** — pruning tables (~7.4 MB) are compiled into the binary. Single-file, no network requests at startup.
- **Dynamic** — tables are passed in as a byte array at runtime, keeping the base WASM bundle under 100 KB.

---

## Benchmarks

100 random scrambles, length 25, parallel solves on Intel Core i7-1255U (10-core, 12th Gen):

| Metric | Result |
|---|---|
| Average solve time | 37.88 ms |
| Table init time | 18.27 ms |
| Success rate | 100/100 |
| Average solution length | 20.5 moves |
| Maximum solution length | 22 moves |

---

## JavaScript / TypeScript

```bash
npm install wasm-rubiks-solver
```

```javascript
import init, { WasmSolver } from './pkg/wasm_rubiks_solver.js';

async function run() {
    await init();

    const solver = WasmSolver.new_embedded();

    const scramble = "R U R' U' F' U2 F";
    const solution = solver.solve(scramble);

    console.log(`Solution: ${solution}`); // e.g. "F U2 F' U R U' R'"
}
```

---

## Rust

```toml
[dependencies]
wasm-rubiks-solver = "0.1.0"
```

```rust
use wasm_rubiks_solver::{CubeState, DominoSolver};

fn main() {
    let solver = DominoSolver::new();

    let mut cube = CubeState::new();
    cube.apply_move(Move::R);
    cube.apply_move(Move::U);
    cube.apply_move(Move::R3); // R'

    if let Some(solution) = solver.solve(&cube) {
        println!("Solution: {:?}", solution);
    }
}
```

---

## Building from source

```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install wasm-pack
cargo install wasm-pack

# Build WASM package (output in ./pkg/)
wasm-pack build --target web --release

# Run benchmarks
cargo run --release -- competition
```

---

## License

MIT. See `Cargo.toml`.

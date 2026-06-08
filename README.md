# wasm-rubiks-solver 🧩⚡

An extremely fast, memory-optimized 2-Phase Rubik's Cube Solver written in Rust and compiled to WebAssembly (WASM). It solves random scrambles directly in the browser or in native Rust environments in **under 40ms** on average, while guaranteeing a solution length of **22 moves or fewer**.

## 🚀 Key Features

* **Hyper-Fast Solving**: Leverages Herbert Kociemba's 2-phase algorithm with advanced D4 symmetry group reductions.
* **Optimized Search**: Eliminates redundant branch checks using precomputed allowed-moves lookup tables and best-first candidate sorting.
* **Browser Ready**: Native timing APIs are emulated (via `Date.now()`) to prevent runtime panics in browser WASM environments.
* **Flexible Initialization**: 
  * **Embedded Mode**: Compiles the precomputed search tables (~7.4 MB) directly into the WASM binary for single-file, zero-fetch startup.
  * **Dynamic Mode**: Accepts search tables as a raw byte array, keeping the base WASM bundle size **under 100 KB** for fast initial page loads.
* **100% Safe Rust**: Built with zero raw `unsafe` code and compiles warning-free under standard Rust compiler and Clippy checks.

---

## 📊 Performance Benchmarks (Native)

Running on 100 random scrambles (length 25, parallel solves):
* **Average Solve Time**: `~37.88 ms`
* **Initialization Time**: `~18.27 ms`
* **Success Rate**: `100% (100/100 solved)`
* **Solution Lengths**: `Average: 20.5 moves, Maximum: 22 moves`

---

## 📦 WebAssembly (JS / TS) Usage

Install the package via npm:
```bash
npm install wasm-rubiks-solver
```

### 💻 Using in the Browser

You can initialize and call the solver using standard ES Modules:

```javascript
import init, { WasmSolver } from './pkg/wasm_rubiks_solver.js';

async function run() {
    // 1. Initialize WASM module loader
    await init();

    // 2. Instantiate the solver (loads embedded pruning tables)
    console.log("Loading solver tables...");
    const solver = WasmSolver.new_embedded();
    console.log("Solver ready!");

    // 3. Solve a scramble (space-separated standard notation)
    const scramble = "R U R' U' F' U2 F";
    const solution = solver.solve(scramble);

    console.log(`Scramble: ${scramble}`);
    console.log(`Solution: ${solution}`); // e.g. "F U2 F' U R U' R'"
}

run();
```

---

## 🦀 Rust Crate Usage

Add `wasm-rubiks-solver` to your `Cargo.toml`:
```toml
[dependencies]
wasm-rubiks-solver = "0.1.0"
```

### 💻 Solving a Cube in Rust

```rust
use wasm_rubiks_solver::{CubeState, DominoSolver};

fn main() {
    // 1. Initialize solver (automatically loads or generates table cache)
    let solver = DominoSolver::new();

    // 2. Setup your scramble
    let mut cube = CubeState::new();
    cube.apply_move(Move::R);
    cube.apply_move(Move::U);
    cube.apply_move(Move::R3); // R'

    // 3. Solve!
    if let Some(solution) = solver.solve(&cube) {
        println!("Solution found: {:?}", solution);
    } else {
        println!("Failed to find solution.");
    }
}
```

---

## 🛠️ Building From Source

### Prerequisites
Make sure you have Rust and `wasm-pack` installed:
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install wasm-pack
cargo install wasm-pack
```

### 1. Compile the WASM Package
To build the WebAssembly binaries and JS bindings:
```bash
wasm-pack build --target web --release
```
The output will be generated inside the `./pkg/` folder.

### 2. Run Native Benchmarks
To run the solver benchmark suite natively:
```bash
cargo run --release -- competition
```

---

## 📄 License
This project is licensed under the **MIT License** - see the `Cargo.toml` file for details.

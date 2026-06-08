use rubiks_cube::{CubeState, DominoSolver, Move, PHASE1_MOVES, SolveStats};
use rayon::prelude::*;
use std::env;
use std::time::{Duration, Instant};

// ---------- Constants ----------

const BENCHMARK_SOLVES: usize = 100;
const BENCHMARK_SCRAMBLE_LEN: usize = 3;
const BENCHMARK_SEED: u64 = 0xC0FFEE;
const BENCHMARK_SOLVE_TIMEOUT_MS: u64 = 500;
const COMPETITION_SOLVES: usize = 100;
const COMPETITION_SCRAMBLE_LEN: usize = 25;
const COMPETITION_SEED: u64 = 0xFACEFEED;
const COMPETITION_SOLVE_TIMEOUT_MS: u64 = 13_116;

#[derive(Debug)]
struct BenchmarkStats {
    solves: usize,
    failures: usize,
    timeouts: usize,
    min_moves: usize,
    max_moves: usize,
    total_moves: usize,
    min_time: Duration,
    max_time: Duration,
    total_time: Duration,
}

impl BenchmarkStats {
    fn new() -> Self {
        Self {
            solves: 0,
            failures: 0,
            timeouts: 0,
            min_moves: usize::MAX,
            max_moves: 0,
            total_moves: 0,
            min_time: Duration::MAX,
            max_time: Duration::ZERO,
            total_time: Duration::ZERO,
        }
    }
    fn record_success(&mut self, moves: usize, t: Duration) {
        self.solves += 1;
        self.min_moves = self.min_moves.min(moves);
        self.max_moves = self.max_moves.max(moves);
        self.total_moves += moves;
        self.min_time = self.min_time.min(t);
        self.max_time = self.max_time.max(t);
        self.total_time += t;
    }
    fn record_failure(&mut self) { self.failures += 1; }
    fn record_timeout(&mut self) { self.timeouts += 1; }
    fn avg_moves(&self) -> f64 { self.total_moves as f64 / self.solves as f64 }
    fn avg_time(&self) -> Duration {
        Duration::from_secs_f64(self.total_time.as_secs_f64() / self.solves as f64)
    }
}

struct SimpleRng {
    state: u64,
}

impl SimpleRng {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }
    fn next_index(&mut self, len: usize) -> usize {
        (self.next_u64() as usize) % len
    }
}

fn main() {
    let mode = env::args().nth(1);
    println!("Initializing Solver and Tables...");
    let init_t = Instant::now();
    let solver = DominoSolver::new();
    println!("Solver initialization took {:.2?}.", init_t.elapsed());

    if mode.as_deref() == Some("competition") {
        run_benchmark(
            "competition",
            &solver,
            COMPETITION_SOLVES,
            COMPETITION_SCRAMBLE_LEN,
            COMPETITION_SEED,
            COMPETITION_SOLVE_TIMEOUT_MS,
        );
        return;
    }

    run_profiled_scramble("basic commutator", &solver, &[Move::R, Move::U, Move::R3, Move::U3]);
    run_profiled_scramble("mixed 6-move", &solver, &[Move::F, Move::R, Move::U, Move::R3, Move::U3, Move::F3]);
    run_profiled_scramble("phase-2 heavy", &solver, &[Move::U, Move::R2, Move::D, Move::L2, Move::U3, Move::F2]);
    run_benchmark(
        "short",
        &solver,
        BENCHMARK_SOLVES,
        BENCHMARK_SCRAMBLE_LEN,
        BENCHMARK_SEED,
        BENCHMARK_SOLVE_TIMEOUT_MS,
    );
}

fn run_profiled_scramble(label: &str, solver: &DominoSolver, scramble: &[Move]) {
    let mut cube = CubeState::new();
    for &m in scramble {
        cube.apply_move(m);
    }
    println!("\nSolving {}: {:?}", label, scramble);
    let (solution, stats) = solver.solve_profiled(&cube);
    if let Some(sol) = solution {
        println!("Solved in {:.2?}.", stats.total_elapsed);
        println!("Moves ({}): {:?}", sol.len(), sol);
        println!(
            "Profile: p1 depth {:?}, {} p1 sols, {} p1 nodes, {} p1 prunes, p1 {:.2?}; \
             {} p2 attempts, {} p2 nodes, p2 {:.2?}.",
            stats.phase1_depth,
            stats.phase1_solutions,
            stats.phase1_nodes,
            stats.phase1_pruned,
            stats.phase1_elapsed,
            stats.phase2_attempts,
            stats.phase2_nodes,
            stats.phase2_elapsed,
        );
        let mut check = cube;
        for &m in &sol {
            check.apply_move(m);
        }
        assert!(check.is_solved(), "Solution produced an invalid state!");
        println!("State verified: 100% Solved.");
    } else {
        println!("Failed. Profile: {:?}", stats);
    }
}

fn run_benchmark(
    label: &str,
    solver: &DominoSolver,
    solves: usize,
    scramble_len: usize,
    seed: u64,
    timeout_ms: u64,
) {
    let mut rng = SimpleRng::new(seed);
    println!(
        "\nRunning {} benchmark: {} scrambles, len {}, seed {:#x}, timeout {}ms, {} threads.",
        label,
        solves,
        scramble_len,
        seed,
        timeout_ms,
        rayon::current_num_threads()
    );

    // Pre-generate all cubes sequentially — RNG must stay deterministic.
    let cubes: Vec<CubeState> = (0..solves)
        .map(|_| {
            let scramble = random_scramble(&mut rng, scramble_len);
            CubeState::new().apply_moves(&scramble)
        })
        .collect();

    let bm_t = Instant::now();

    // Solve all scrambles in parallel.
    let results: Vec<(Option<Vec<Move>>, SolveStats)> = cubes
        .par_iter()
        .map(|cube| solver.solve_profiled_with_time_limit(cube, Some(Duration::from_millis(timeout_ms))))
        .collect();

    // Aggregate results (sequential, order preserved by rayon's collect).
    let mut stats = BenchmarkStats::new();
    for (i, (sol, ss)) in results.iter().enumerate() {
        if let Some(sol) = sol {
            assert!(cubes[i].apply_moves(sol).is_solved(), "Solve {} invalid!", i + 1);
            stats.record_success(sol.len(), ss.total_elapsed);
        } else if ss.timed_out {
            stats.record_timeout();
        } else {
            stats.record_failure();
        }

        if (i + 1) % 10 == 0 || i + 1 == solves {
            if stats.solves > 0 {
                println!(
                    "  completed {}/{} solves; avg {:.2} moves, avg {:.2?}.",
                    i + 1,
                    solves,
                    stats.avg_moves(),
                    stats.avg_time()
                );
            } else {
                println!("  completed {}/{} solves; no successful solves yet.", i + 1, solves);
            }
        }
    }

    println!(
        "Benchmark completed in {:.2?} wall clock ({} threads).",
        bm_t.elapsed(),
        rayon::current_num_threads()
    );
    println!("Successful solves: {}/{}.", stats.solves, solves);
    if stats.solves > 0 {
        println!("Moves: min {}, avg {:.2}, max {}.", stats.min_moves, stats.avg_moves(), stats.max_moves);
        println!(
            "Time (per-solve): min {:.2?}, avg {:.2?}, max {:.2?}.",
            stats.min_time,
            stats.avg_time(),
            stats.max_time
        );
    }
    if stats.failures > 0 {
        println!("Failures: {}.", stats.failures);
    }
    if stats.timeouts > 0 {
        println!("Timeouts: {}.", stats.timeouts);
    }
}

fn random_scramble(rng: &mut SimpleRng, len: usize) -> Vec<Move> {
    let mut s = Vec::with_capacity(len);
    while s.len() < len {
        let next = PHASE1_MOVES[rng.next_index(PHASE1_MOVES.len())];
        if !is_redundant(&s, next) {
            s.push(next);
        }
    }
    s
}

fn is_redundant(path: &[Move], next: Move) -> bool {
    if let Some(&last) = path.last() {
        (last as u8 / 3) == (next as u8 / 3)
    } else {
        false
    }
}

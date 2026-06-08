use rayon::prelude::*;
use std::collections::VecDeque;
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
const PHASE1_EXTRA_OPTIMIZATION_DEPTH: usize = 1;

const EDGE_ORIENTATION_COUNT: usize = 2048;    // 2^11
const CORNER_ORIENTATION_COUNT: usize = 2187;  // 3^7
const SLICE_COMBINATION_COUNT: usize = 495;    // C(12,4)
const CORNER_PERMUTATION_COUNT: usize = 40320; // 8!
const UD_EDGE_PERMUTATION_COUNT: usize = 40320; // 8!
const SLICE_EDGE_PERMUTATION_COUNT: usize = 24; // 4!
const PHASE1_MOVE_COUNT: usize = 18;
const PHASE2_MOVE_COUNT: usize = 10;
const PRUNE_UNKNOWN: u8 = u8::MAX;

const TABLE_CACHE_PATH: &str = "pruning_tables.bin";
/// Bump whenever table layout or semantics change to force regeneration.
const TABLE_MAGIC: &[u8; 12] = b"RUBIK2PH_V4\0";
/// Combined Phase-2 edge prune table size: ud (40320) × slice_ep (24).
const PHASE2_UD_SLICE_EP_SIZE: usize = UD_EDGE_PERMUTATION_COUNT * SLICE_EDGE_PERMUTATION_COUNT;
/// Combined Phase-2 corner prune table size: cp (40320) × slice_ep (24).
const PHASE2_CP_SEP_SIZE: usize = CORNER_PERMUTATION_COUNT * SLICE_EDGE_PERMUTATION_COUNT;

/// rank({8,9,10,11}) in C(12,4) — the slice coord value for the solved state.
const SOLVED_SLICE_COORD: u16 = 494;

/// Face index per move: U=0, D=1, F=2, B=3, L=4, R=5.  axis = face / 2.
const PHASE1_MOVE_FACES: [u8; PHASE1_MOVE_COUNT] = [
    0, 0, 0,  // U, U2, U3
    1, 1, 1,  // D, D2, D3
    2, 2, 2,  // F, F2, F3
    3, 3, 3,  // B, B2, B3
    4, 4, 4,  // L, L2, L3
    5, 5, 5,  // R, R2, R3
];
const PHASE2_MOVE_FACES: [u8; PHASE2_MOVE_COUNT] = [
    0, 0, 0,  // U, U2, U3
    1, 1, 1,  // D, D2, D3
    2,        // F2
    3,        // B2
    4,        // L2
    5,        // R2
];

// ---------- 1. Move Enum & Arrays ----------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Move {
    U, U2, U3, D, D2, D3,
    F, F2, F3, B, B2, B3,
    L, L2, L3, R, R2, R3,
}

const PHASE1_MOVES: [Move; PHASE1_MOVE_COUNT] = [
    Move::U, Move::U2, Move::U3, Move::D, Move::D2, Move::D3,
    Move::F, Move::F2, Move::F3, Move::B, Move::B2, Move::B3,
    Move::L, Move::L2, Move::L3, Move::R, Move::R2, Move::R3,
];

const PHASE2_MOVES: [Move; PHASE2_MOVE_COUNT] = [
    Move::U, Move::U2, Move::U3, Move::D, Move::D2, Move::D3,
    Move::F2, Move::B2, Move::L2, Move::R2,
];

// ---------- 2. CubeState ----------

/// Highly optimized byte-array representation.
/// Move application is heavily vectorized by the compiler using these small arrays.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct CubeState {
    pub ep: [u8; 12], // Edge permutations
    pub eo: [u8; 12], // Edge orientations (0..1)
    pub cp: [u8; 8],  // Corner permutations
    pub co: [u8; 8],  // Corner orientations (0..2)
}

impl Default for CubeState {
    fn default() -> Self {
        Self::new()
    }
}

impl CubeState {
    pub fn new() -> Self {
        Self {
            ep: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            eo: [0; 12],
            cp: [0, 1, 2, 3, 4, 5, 6, 7],
            co: [0; 8],
        }
    }

    pub fn is_solved(&self) -> bool { *self == Self::new() }

    pub fn edge_orientation_coord(&self) -> usize {
        let mut coord = 0;
        for (i, &eo_val) in self.eo.iter().take(11).enumerate() {
            coord |= (eo_val as usize) << i;
        }
        coord
    }

    pub fn corner_orientation_coord(&self) -> usize {
        let mut coord = 0;
        for &co_val in self.co.iter().take(7) {
            coord = coord * 3 + co_val as usize;
        }
        coord
    }

    pub fn slice_coord(&self) -> usize {
        let mut selected = [0usize; 4];
        let mut count = 0;
        for (i, &ep_val) in self.ep.iter().enumerate() {
            if ep_val >= 8 {
                selected[count] = i;
                count += 1;
            }
        }
        debug_assert_eq!(count, 4);
        let mut rank = 0;
        let mut previous = 0;
        let mut remaining = 4;
        for (idx, &position) in selected.iter().enumerate() {
            let start = if idx == 0 { 0 } else { previous + 1 };
            for x in start..position { rank += binomial(12 - x - 1, remaining - 1); }
            previous = position;
            remaining -= 1;
        }
        rank
    }

    pub fn corner_permutation_coord(&self) -> usize { permutation_rank(&self.cp) }

    pub fn ud_edge_permutation_coord(&self) -> usize {
        let mut edges = [0u8; 8];
        for (i, val) in edges.iter_mut().enumerate() {
            debug_assert!(self.ep[i] < 8);
            *val = self.ep[i];
        }
        permutation_rank(&edges)
    }

    pub fn slice_edge_permutation_coord(&self) -> usize {
        let mut edges = [0u8; 4];
        for (i, val) in edges.iter_mut().enumerate() {
            debug_assert!(self.ep[8 + i] >= 8);
            *val = self.ep[8 + i] - 8;
        }
        permutation_rank(&edges)
    }

    pub fn apply_state(&mut self, m: &CubeState) {
        let mut next_ep = [0u8; 12];
        let mut next_eo = [0u8; 12];
        for i in 0..12 {
            next_ep[i] = self.ep[m.ep[i] as usize];
            next_eo[i] = (self.eo[m.ep[i] as usize] + m.eo[i]) % 2;
        }
        let mut next_cp = [0u8; 8];
        let mut next_co = [0u8; 8];
        for i in 0..8 {
            next_cp[i] = self.cp[m.cp[i] as usize];
            next_co[i] = (self.co[m.cp[i] as usize] + m.co[i]) % 3;
        }
        self.ep = next_ep; self.eo = next_eo;
        self.cp = next_cp; self.co = next_co;
    }


    pub fn apply_move(&mut self, m: Move) {
        let base = match m {
            Move::U | Move::U2 | Move::U3 => BASE_MOVES[0],
            Move::D | Move::D2 | Move::D3 => BASE_MOVES[1],
            Move::F | Move::F2 | Move::F3 => BASE_MOVES[2],
            Move::B | Move::B2 | Move::B3 => BASE_MOVES[3],
            Move::L | Move::L2 | Move::L3 => BASE_MOVES[4],
            Move::R | Move::R2 | Move::R3 => BASE_MOVES[5],
        };
        self.apply_state(&base);
        let power = match m {
            Move::U2|Move::D2|Move::F2|Move::B2|Move::L2|Move::R2 => 2,
            Move::U3|Move::D3|Move::F3|Move::B3|Move::L3|Move::R3 => 3,
            _ => 1,
        };
        for _ in 1..power { self.apply_state(&base); }
    }

    pub fn apply_moves(&self, moves: &[Move]) -> Self {
        let mut s = *self;
        for &m in moves { s.apply_move(m); }
        s
    }
}

// ---------- 3. Coordinate Decoders ----------

#[allow(clippy::too_many_arguments)]
/// Decodes an EO coordinate (0..2048) → 12-element orientation array.
fn decode_eo_coord(coord: usize) -> [u8; 12] {
    let mut eo = [0u8; 12];
    let mut parity = 0usize;
    for (i, val) in eo.iter_mut().take(11).enumerate() {
        *val = ((coord >> i) & 1) as u8;
        parity += *val as usize;
    }
    eo[11] = (parity % 2) as u8;
    eo
}

#[allow(clippy::too_many_arguments)]
/// Decodes a CO coordinate (0..2187) → 8-element orientation array.
fn decode_co_coord(coord: usize) -> [u8; 8] {
    let mut co = [0u8; 8];
    let mut remaining = coord;
    let mut sum = 0usize;
    for (i, val) in co.iter_mut().take(7).enumerate() {
        let d = 3usize.pow((6 - i) as u32);
        *val = (remaining / d) as u8;
        remaining %= d;
        sum += *val as usize;
    }
    co[7] = ((3 - sum % 3) % 3) as u8;
    co
}

#[allow(clippy::too_many_arguments)]
/// Decodes a slice combination rank (0..495) → which of the 12 positions hold slice edges.
fn decode_slice_coord(mut rank: usize) -> [bool; 12] {
    let mut is_slice = [false; 12];
    let mut remaining_k = 4usize;
    let mut x = 0usize;
    while remaining_k > 0 && x < 12 {
        let c = binomial(12 - x - 1, remaining_k - 1);
        if rank < c { is_slice[x] = true; remaining_k -= 1; }
        else { rank -= c; }
        x += 1;
    }
    is_slice
}

/// Decodes a Lehmer-code rank → permutation of 0..N.
fn decode_permutation<const N: usize>(mut rank: usize) -> [u8; N] {
    let mut perm = [0u8; N];
    let mut used = [false; N];
    for (i, p_val) in perm.iter_mut().enumerate() {
        let f = factorial(N - i - 1);
        let mut k = rank / f;
        rank %= f;
        for (j, u_val) in used.iter_mut().enumerate() {
            if !*u_val {
                if k == 0 { *p_val = j as u8; *u_val = true; break; }
                k -= 1;
            }
        }
    }
    perm
}

// ---------- 4. Move Table Builders ----------

/// `eo_move[eo][mi]` → new EO coord after Phase-1 move mi.
fn build_eo_move_table() -> Vec<[u16; PHASE1_MOVE_COUNT]> {
    let mut t = vec![[0u16; PHASE1_MOVE_COUNT]; EDGE_ORIENTATION_COUNT];
    for (coord, row) in t.iter_mut().enumerate() {
        let mut base = CubeState::new();
        base.eo = decode_eo_coord(coord);
        for (mi, &mv) in PHASE1_MOVES.iter().enumerate() {
            let mut s = base; s.apply_move(mv);
            row[mi] = s.edge_orientation_coord() as u16;
        }
    }
    t
}

/// `co_move[co][mi]` → new CO coord after Phase-1 move mi.
fn build_co_move_table() -> Vec<[u16; PHASE1_MOVE_COUNT]> {
    let mut t = vec![[0u16; PHASE1_MOVE_COUNT]; CORNER_ORIENTATION_COUNT];
    for (coord, row) in t.iter_mut().enumerate() {
        let mut base = CubeState::new();
        base.co = decode_co_coord(coord);
        for (mi, &mv) in PHASE1_MOVES.iter().enumerate() {
            let mut s = base; s.apply_move(mv);
            row[mi] = s.corner_orientation_coord() as u16;
        }
    }
    t
}

/// `slice_move[slice][mi]` → new slice coord after Phase-1 move mi.
fn build_slice_move_table() -> Vec<[u16; PHASE1_MOVE_COUNT]> {
    let mut t = vec![[0u16; PHASE1_MOVE_COUNT]; SLICE_COMBINATION_COUNT];
    for (coord, row) in t.iter_mut().enumerate() {
        let is_slice = decode_slice_coord(coord);
        let mut base = CubeState::new();
        let (mut sv, mut uv) = (8u8, 0u8);
        for (i, &slice_flag) in is_slice.iter().enumerate() {
            if slice_flag { base.ep[i] = sv; sv += 1; }
            else           { base.ep[i] = uv; uv += 1; }
        }
        for (mi, &mv) in PHASE1_MOVES.iter().enumerate() {
            let mut s = base; s.apply_move(mv);
            row[mi] = s.slice_coord() as u16;
        }
    }
    t
}

/// `cp_move[cp][mi]` → new corner-perm coord after Phase-2 move mi.
fn build_cp_move_table() -> Vec<[u32; PHASE2_MOVE_COUNT]> {
    let mut t = vec![[0u32; PHASE2_MOVE_COUNT]; CORNER_PERMUTATION_COUNT];
    for (coord, row) in t.iter_mut().enumerate() {
        let mut base = CubeState::new();
        base.cp = decode_permutation::<8>(coord);
        for (mi, &mv) in PHASE2_MOVES.iter().enumerate() {
            let mut s = base; s.apply_move(mv);
            row[mi] = s.corner_permutation_coord() as u32;
        }
    }
    t
}

/// `ud_edge_move[ud][mi]` → new UD-edge-perm coord after Phase-2 move mi.
fn build_ud_edge_move_table() -> Vec<[u32; PHASE2_MOVE_COUNT]> {
    let mut t = vec![[0u32; PHASE2_MOVE_COUNT]; UD_EDGE_PERMUTATION_COUNT];
    for (coord, row) in t.iter_mut().enumerate() {
        let mut base = CubeState::new();
        let ep8 = decode_permutation::<8>(coord);
        base.ep[..8].copy_from_slice(&ep8);
        // ep[8..12] = [8,9,10,11] from new() — valid domino slice layer
        for (mi, &mv) in PHASE2_MOVES.iter().enumerate() {
            let mut s = base; s.apply_move(mv);
            row[mi] = s.ud_edge_permutation_coord() as u32;
        }
    }
    t
}

/// `slice_ep_move[sep][mi]` → new slice-edge-perm coord after Phase-2 move mi.
fn build_slice_ep_move_table() -> Vec<[u8; PHASE2_MOVE_COUNT]> {
    let mut t = vec![[0u8; PHASE2_MOVE_COUNT]; SLICE_EDGE_PERMUTATION_COUNT];
    for (coord, row) in t.iter_mut().enumerate() {
        let mut base = CubeState::new();
        let ep4 = decode_permutation::<4>(coord);
        for (i, &val) in ep4.iter().enumerate() { base.ep[8 + i] = 8 + val; }
        for (mi, &mv) in PHASE2_MOVES.iter().enumerate() {
            let mut s = base; s.apply_move(mv);
            row[mi] = s.slice_edge_permutation_coord() as u8;
        }
    }
    t
}



// ---------- 5. Solver ----------

pub struct DominoSolver {
    // Pruning tables (heuristics)
    phase1_eo_slice_prune:  Vec<u8>,
    phase1_co_slice_prune:  Vec<u8>,
    phase2_ud_slice_ep_prune: Vec<u8>,
    /// Joint (cp × slice_ep) prune table — ~945 KB, replaces scalar corner prune.
    phase2_cp_sep_prune:      Vec<u8>,
    // Coordinate move tables – Phase 1 (all 18 moves)
    eo_move:    Vec<[u16; PHASE1_MOVE_COUNT]>,
    co_move:    Vec<[u16; PHASE1_MOVE_COUNT]>,
    slice_move: Vec<[u16; PHASE1_MOVE_COUNT]>,
    // Coordinate move tables – Phase 2 (10 restricted moves)
    cp_move:       Vec<[u32; PHASE2_MOVE_COUNT]>,
    ud_edge_move:  Vec<[u32; PHASE2_MOVE_COUNT]>,
    slice_ep_move: Vec<[u8;  PHASE2_MOVE_COUNT]>,
}

#[derive(Debug, Default)]
pub struct TableStats { entries: usize, generated_states: usize, elapsed: Duration }

#[derive(Debug, Default)]
pub struct SolveStats {
    phase1_depth: Option<usize>,
    phase1_solutions: usize,
    phase1_nodes: u64,
    phase1_pruned: u64,
    phase1_elapsed: Duration,
    phase2_attempts: usize,
    phase2_nodes: u64,
    phase2_elapsed: Duration,
    total_elapsed: Duration,
    timed_out: bool,
}

impl Default for DominoSolver {
    fn default() -> Self {
        Self::new()
    }
}

impl DominoSolver {
    pub fn new() -> Self {
        let mut solver = Self {
            phase1_eo_slice_prune:  vec![PRUNE_UNKNOWN; EDGE_ORIENTATION_COUNT * SLICE_COMBINATION_COUNT],
            phase1_co_slice_prune:  vec![PRUNE_UNKNOWN; CORNER_ORIENTATION_COUNT * SLICE_COMBINATION_COUNT],
            phase2_ud_slice_ep_prune: vec![PRUNE_UNKNOWN; PHASE2_UD_SLICE_EP_SIZE],
            phase2_cp_sep_prune:      vec![PRUNE_UNKNOWN; PHASE2_CP_SEP_SIZE],
            eo_move:    vec![[0u16; PHASE1_MOVE_COUNT]; EDGE_ORIENTATION_COUNT],
            co_move:    vec![[0u16; PHASE1_MOVE_COUNT]; CORNER_ORIENTATION_COUNT],
            slice_move: vec![[0u16; PHASE1_MOVE_COUNT]; SLICE_COMBINATION_COUNT],
            cp_move:       vec![[0u32; PHASE2_MOVE_COUNT]; CORNER_PERMUTATION_COUNT],
            ud_edge_move:  vec![[0u32; PHASE2_MOVE_COUNT]; UD_EDGE_PERMUTATION_COUNT],
            slice_ep_move: vec![[0u8;  PHASE2_MOVE_COUNT]; SLICE_EDGE_PERMUTATION_COUNT],
        };

        if solver.try_load_tables_from_disk() {
            println!("Loaded all tables from disk cache ({}).", TABLE_CACHE_PATH);
        } else {
            println!("Cache not found or invalid — generating tables...");

            let t0 = Instant::now();
            solver.eo_move    = build_eo_move_table();
            solver.co_move    = build_co_move_table();
            solver.slice_move = build_slice_move_table();
            solver.cp_move       = build_cp_move_table();
            solver.ud_edge_move  = build_ud_edge_move_table();
            solver.slice_ep_move = build_slice_ep_move_table();
            println!("Built coordinate move tables in {:.2?}.", t0.elapsed());

            let ts = solver.generate_prune_tables();
            println!(
                "Built pruning tables: {} entries, {} BFS expansions, {:.2?}.",
                ts.entries, ts.generated_states, ts.elapsed
            );

            let t2 = Instant::now();
            solver.phase2_ud_slice_ep_prune =
                build_ud_slice_ep_prune_table(&solver.ud_edge_move, &solver.slice_ep_move);
            solver.phase2_cp_sep_prune =
                build_cp_sep_prune_table(&solver.cp_move, &solver.slice_ep_move);
            println!("Built combined Phase-2 prune tables ({} + {} entries) in {:.2?}.",
                PHASE2_UD_SLICE_EP_SIZE, PHASE2_CP_SEP_SIZE, t2.elapsed());

            match solver.save_tables_to_disk() {
                Ok(()) => println!("All tables saved to disk ({}).", TABLE_CACHE_PATH),
                Err(e) => eprintln!("Warning: failed to save tables: {e}"),
            }
        }
        solver
    }

    // ---- Disk cache load/save ----

    fn try_load_tables_from_disk(&mut self) -> bool {
        let data = match std::fs::read(TABLE_CACHE_PATH) {
            Ok(d) => d,
            Err(_) => return false,
        };
        let prune_sz = EDGE_ORIENTATION_COUNT * SLICE_COMBINATION_COUNT
            + CORNER_ORIENTATION_COUNT * SLICE_COMBINATION_COUNT
            + PHASE2_UD_SLICE_EP_SIZE
            + PHASE2_CP_SEP_SIZE;
        let move_sz =
            EDGE_ORIENTATION_COUNT   * PHASE1_MOVE_COUNT * 2
            + CORNER_ORIENTATION_COUNT * PHASE1_MOVE_COUNT * 2
            + SLICE_COMBINATION_COUNT  * PHASE1_MOVE_COUNT * 2
            + CORNER_PERMUTATION_COUNT * PHASE2_MOVE_COUNT * 4
            + UD_EDGE_PERMUTATION_COUNT * PHASE2_MOVE_COUNT * 4
            + SLICE_EDGE_PERMUTATION_COUNT * PHASE2_MOVE_COUNT;
        let expected = TABLE_MAGIC.len() + prune_sz + move_sz;

        if data.len() != expected || &data[..TABLE_MAGIC.len()] != TABLE_MAGIC {
            eprintln!("Warning: table cache stale or corrupted — regenerating.");
            return false;
        }

        let mut off = TABLE_MAGIC.len();

        // Pruning tables (u8 slices)
        let n = EDGE_ORIENTATION_COUNT * SLICE_COMBINATION_COUNT;
        self.phase1_eo_slice_prune.copy_from_slice(&data[off..off+n]); off += n;
        let n = CORNER_ORIENTATION_COUNT * SLICE_COMBINATION_COUNT;
        self.phase1_co_slice_prune.copy_from_slice(&data[off..off+n]); off += n;
        self.phase2_ud_slice_ep_prune.copy_from_slice(&data[off..off+PHASE2_UD_SLICE_EP_SIZE]);
        off += PHASE2_UD_SLICE_EP_SIZE;
        self.phase2_cp_sep_prune.copy_from_slice(&data[off..off+PHASE2_CP_SEP_SIZE]);
        off += PHASE2_CP_SEP_SIZE;

        // Move tables
        self.eo_move    = read_u16_table::<PHASE1_MOVE_COUNT>(&data, &mut off, EDGE_ORIENTATION_COUNT);
        self.co_move    = read_u16_table::<PHASE1_MOVE_COUNT>(&data, &mut off, CORNER_ORIENTATION_COUNT);
        self.slice_move = read_u16_table::<PHASE1_MOVE_COUNT>(&data, &mut off, SLICE_COMBINATION_COUNT);
        self.cp_move      = read_u32_table::<PHASE2_MOVE_COUNT>(&data, &mut off, CORNER_PERMUTATION_COUNT);
        self.ud_edge_move = read_u32_table::<PHASE2_MOVE_COUNT>(&data, &mut off, UD_EDGE_PERMUTATION_COUNT);
        for row in &mut self.slice_ep_move {
            row.copy_from_slice(&data[off..off + PHASE2_MOVE_COUNT]);
            off += PHASE2_MOVE_COUNT;
        }
        let _ = off; // last increment is intentionally dead

        true
    }

    fn save_tables_to_disk(&self) -> std::io::Result<()> {
        let mut data = Vec::new();
        data.extend_from_slice(TABLE_MAGIC);
        data.extend_from_slice(&self.phase1_eo_slice_prune);
        data.extend_from_slice(&self.phase1_co_slice_prune);
        data.extend_from_slice(&self.phase2_ud_slice_ep_prune);
        data.extend_from_slice(&self.phase2_cp_sep_prune);
        write_u16_table(&mut data, &self.eo_move);
        write_u16_table(&mut data, &self.co_move);
        write_u16_table(&mut data, &self.slice_move);
        write_u32_table(&mut data, &self.cp_move);
        write_u32_table(&mut data, &self.ud_edge_move);
        for row in &self.slice_ep_move { data.extend_from_slice(row); }
        std::fs::write(TABLE_CACHE_PATH, &data)
    }

    fn generate_prune_tables(&mut self) -> TableStats {
        let t0 = Instant::now();
        let s0 = fill_phase1_prune_table(&mut self.phase1_eo_slice_prune, Phase1PruneKind::EdgeOrientationSlice);
        let s1 = fill_phase1_prune_table(&mut self.phase1_co_slice_prune, Phase1PruneKind::CornerOrientationSlice);
        TableStats {
            entries: s0.0 + s1.0,
            generated_states: s0.1 + s1.1,
            elapsed: t0.elapsed(),
        }
    }

    // ---- Heuristics (coord-based, no CubeState) ----

    #[inline]
    fn phase1_heuristic(&self, eo: u16, co: u16, slice: u16) -> usize {
        let eo_d = self.phase1_eo_slice_prune[eo as usize * SLICE_COMBINATION_COUNT + slice as usize];
        let co_d = self.phase1_co_slice_prune[co as usize * SLICE_COMBINATION_COUNT + slice as usize];
        eo_d.max(co_d) as usize
    }

    #[inline]
    fn phase2_heuristic(&self, cp: u32, ud: u32, sep: u8) -> usize {
        let edge_h  = self.phase2_ud_slice_ep_prune[
            ud as usize * SLICE_EDGE_PERMUTATION_COUNT + sep as usize];
        let corner_h = self.phase2_cp_sep_prune[
            cp as usize * SLICE_EDGE_PERMUTATION_COUNT + sep as usize];
        edge_h.max(corner_h) as usize
    }

    // ---- Public solve API ----

    pub fn solve(&self, s: &CubeState) -> Option<Vec<Move>> { self.solve_profiled(s).0 }

    pub fn solve_profiled(&self, s: &CubeState) -> (Option<Vec<Move>>, SolveStats) {
        self.solve_profiled_with_time_limit(s, None)
    }

    pub fn solve_profiled_with_time_limit(
        &self, start: &CubeState, limit: Option<Duration>,
    ) -> (Option<Vec<Move>>, SolveStats) {
        let t0 = Instant::now();
        let max_p1 = 12;
        let max_p2 = 18;
        let mut stats = SolveStats::default();
        let mut best: Option<Vec<Move>> = None;
        let mut best_len = usize::MAX;

        // Compute Phase-1 coordinates once from the start state.
        let init_eo    = start.edge_orientation_coord() as u16;
        let init_co    = start.corner_orientation_coord() as u16;
        let init_slice = start.slice_coord() as u16;

        let p1_t = Instant::now();
        let p2_t = Instant::now();
        let mut deepest_p1 = max_p1;

        for depth in 0..=max_p1 {
            if depth > deepest_p1 || depth >= best_len { break; }

            let mut p1_sols: Vec<Vec<usize>> = Vec::new();
            self.search_phase1(
                init_eo, init_co, init_slice, depth, 0,
                &mut Vec::new(), &mut p1_sols, &mut stats, t0, limit,
            );
            if stats.timed_out { break; }
            if p1_sols.is_empty() { continue; }

            if stats.phase1_depth.is_none() {
                stats.phase1_depth = Some(depth);
                deepest_p1 = (depth + PHASE1_EXTRA_OPTIMIZATION_DEPTH).min(max_p1);
            }
            stats.phase1_solutions += p1_sols.len();

            for p1_idx in p1_sols {
                if stats.timed_out { break; }

                // Reconstruct domino CubeState to extract Phase-2 coords.
                let p1_moves: Vec<Move> = p1_idx.iter().map(|&i| PHASE1_MOVES[i]).collect();
                let domino = start.apply_moves(&p1_moves);
                let init_cp  = domino.corner_permutation_coord() as u32;
                let init_ud  = domino.ud_edge_permutation_coord() as u32;
                let init_sep = domino.slice_edge_permutation_coord() as u8;

                let max_p2_here = max_p2.min(best_len.saturating_sub(p1_moves.len() + 1));
                if let Some(p2_idx) = self.search_phase2(
                    init_cp, init_ud, init_sep, max_p2_here, &mut stats, t0, limit,
                ) {
                    let p2_moves: Vec<Move> = p2_idx.iter().map(|&i| PHASE2_MOVES[i]).collect();
                    let mut full = p1_moves.clone();
                    full.extend(p2_moves);
                    full = normalize_moves(&full);
                    if full.len() < best_len { best_len = full.len(); best = Some(full); }
                }
            }
        }
        stats.phase1_elapsed = p1_t.elapsed();
        stats.phase2_elapsed = p2_t.elapsed();
        stats.total_elapsed  = t0.elapsed();
        (best, stats)
    }

    // ---- Phase 1 IDA* — operates purely on (eo, co, slice) coordinates ----

    #[allow(clippy::too_many_arguments)]
    fn search_phase1(
        &self, eo: u16, co: u16, slice: u16,
        max_d: usize, d: usize,
        path: &mut Vec<usize>, sols: &mut Vec<Vec<usize>>,
        stats: &mut SolveStats, t0: Instant, limit: Option<Duration>,
    ) {
        stats.phase1_nodes += 1;
        if should_stop_search(stats, t0, limit) { return; }

        if d == max_d {
            if eo == 0 && co == 0 && slice == SOLVED_SLICE_COORD { sols.push(path.clone()); }
            return;
        }

        let h = self.phase1_heuristic(eo, co, slice);
        if d + h > max_d { stats.phase1_pruned += 1; return; }

        for mi in 0..PHASE1_MOVE_COUNT {
            if stats.timed_out { return; }
            if is_redundant_p1(path, mi) { continue; }
            let neo   = self.eo_move[eo as usize][mi];
            let nco   = self.co_move[co as usize][mi];
            let nsl   = self.slice_move[slice as usize][mi];
            path.push(mi);
            self.search_phase1(neo, nco, nsl, max_d, d + 1, path, sols, stats, t0, limit);
            path.pop();
        }
    }

    // ---- Phase 2 IDA* — operates purely on (cp, ud, sep) coordinates ----

    #[allow(clippy::too_many_arguments)]
    fn search_phase2(
        &self, cp: u32, ud: u32, sep: u8,
        max_d: usize, stats: &mut SolveStats, t0: Instant, limit: Option<Duration>,
    ) -> Option<Vec<usize>> {
        stats.phase2_attempts += 1;
        for depth in 0..=max_d {
            if stats.timed_out { return None; }
            let mut path = Vec::new();
            if self.do_phase2(cp, ud, sep, depth, 0, &mut path, stats, t0, limit) {
                return Some(path);
            }
        }
        None
    }

    #[allow(clippy::too_many_arguments)]
    fn do_phase2(
        &self, cp: u32, ud: u32, sep: u8,
        max_d: usize, d: usize,
        path: &mut Vec<usize>, stats: &mut SolveStats, t0: Instant, limit: Option<Duration>,
    ) -> bool {
        stats.phase2_nodes += 1;
        if should_stop_search(stats, t0, limit) { return false; }
        let h = self.phase2_heuristic(cp, ud, sep);
        if d + h > max_d { return false; }
        if d == max_d { return cp == 0 && ud == 0 && sep == 0; }

        for mi in 0..PHASE2_MOVE_COUNT {
            if stats.timed_out { return false; }
            if is_redundant_p2(path, mi) { continue; }
            let ncp  = self.cp_move[cp as usize][mi];
            let nud  = self.ud_edge_move[ud as usize][mi];
            let nsep = self.slice_ep_move[sep as usize][mi];
            path.push(mi);
            if self.do_phase2(ncp, nud, nsep, max_d, d + 1, path, stats, t0, limit) { return true; }
            path.pop();
        }
        false
    }
}

// ---------- 6. Table I/O helpers ----------

fn write_u16_table<const N: usize>(buf: &mut Vec<u8>, t: &[[u16; N]]) {
    for row in t { for &v in row { buf.extend_from_slice(&v.to_le_bytes()); } }
}
fn write_u32_table<const N: usize>(buf: &mut Vec<u8>, t: &[[u32; N]]) {
    for row in t { for &v in row { buf.extend_from_slice(&v.to_le_bytes()); } }
}
fn read_u16_table<const K: usize>(data: &[u8], off: &mut usize, rows: usize) -> Vec<[u16; K]> {
    let mut t = vec![[0u16; K]; rows];
    for row in &mut t {
        for v in row.iter_mut() {
            *v = u16::from_le_bytes(data[*off..*off+2].try_into().unwrap());
            *off += 2;
        }
    }
    t
}
fn read_u32_table<const K: usize>(data: &[u8], off: &mut usize, rows: usize) -> Vec<[u32; K]> {
    let mut t = vec![[0u32; K]; rows];
    for row in &mut t {
        for v in row.iter_mut() {
            *v = u32::from_le_bytes(data[*off..*off+4].try_into().unwrap());
            *off += 4;
        }
    }
    t
}

// ---------- 7. Pruning-table BFS (unchanged logic) ----------

#[derive(Clone, Copy)]
enum Phase1PruneKind { EdgeOrientationSlice, CornerOrientationSlice }
/// Builds the combined (ud_edge × slice_ep) Phase-2 prune table using move-table BFS.
/// No CubeState needed — pure coordinate-space traversal.
fn build_ud_slice_ep_prune_table(
    ud_move: &[[u32; PHASE2_MOVE_COUNT]],
    sep_move: &[[u8; PHASE2_MOVE_COUNT]],
) -> Vec<u8> {
    let mut table = vec![PRUNE_UNKNOWN; PHASE2_UD_SLICE_EP_SIZE];
    let mut queue: VecDeque<(u32, u8)> = VecDeque::new();
    // Solved state: ud=0, sep=0
    table[0] = 0;
    queue.push_back((0, 0));
    while let Some((ud, sep)) = queue.pop_front() {
        let depth = table[ud as usize * SLICE_EDGE_PERMUTATION_COUNT + sep as usize];
        for mi in 0..PHASE2_MOVE_COUNT {
            let nud  = ud_move[ud as usize][mi];
            let nsep = sep_move[sep as usize][mi];
            let key  = nud as usize * SLICE_EDGE_PERMUTATION_COUNT + nsep as usize;
            if table[key] == PRUNE_UNKNOWN {
                table[key] = depth + 1;
                queue.push_back((nud, nsep));
            }
        }
    }
    table
}

/// Builds the combined (cp × slice_ep) Phase-2 prune table using move-table BFS.
fn build_cp_sep_prune_table(
    cp_move: &[[u32; PHASE2_MOVE_COUNT]],
    sep_move: &[[u8;  PHASE2_MOVE_COUNT]],
) -> Vec<u8> {
    let mut table = vec![PRUNE_UNKNOWN; PHASE2_CP_SEP_SIZE];
    let mut queue: VecDeque<(u32, u8)> = VecDeque::new();
    table[0] = 0;
    queue.push_back((0, 0));
    while let Some((cp, sep)) = queue.pop_front() {
        let depth = table[cp as usize * SLICE_EDGE_PERMUTATION_COUNT + sep as usize];
        for mi in 0..PHASE2_MOVE_COUNT {
            let ncp  = cp_move[cp as usize][mi];
            let nsep = sep_move[sep as usize][mi];
            let key  = ncp as usize * SLICE_EDGE_PERMUTATION_COUNT + nsep as usize;
            if table[key] == PRUNE_UNKNOWN {
                table[key] = depth + 1;
                queue.push_back((ncp, nsep));
            }
        }
    }
    table
}

fn fill_phase1_prune_table(table: &mut [u8], kind: Phase1PruneKind) -> (usize, usize) {
    let mut q = VecDeque::new();
    let solved = CubeState::new();
    let sk = phase1_pair_key(&solved, kind);
    table[sk] = 0; q.push_back(solved);
    let mut filled = 1; let mut generated = 0;
    while let Some(state) = q.pop_front() {
        let depth = table[phase1_pair_key(&state, kind)];
        for &m in &PHASE1_MOVES {
            let mut next = state; next.apply_move(m); generated += 1;
            let k = phase1_pair_key(&next, kind);
            if table[k] == PRUNE_UNKNOWN { table[k] = depth+1; filled += 1; q.push_back(next); }
        }
    }
    (filled, generated)
}

fn phase1_pair_key(s: &CubeState, kind: Phase1PruneKind) -> usize {
    let sl = s.slice_coord();
    match kind {
        Phase1PruneKind::EdgeOrientationSlice   => s.edge_orientation_coord()   * SLICE_COMBINATION_COUNT + sl,
        Phase1PruneKind::CornerOrientationSlice => s.corner_orientation_coord() * SLICE_COMBINATION_COUNT + sl,
    }
}



// ---------- 8. Utilities ----------

fn permutation_rank<const N: usize>(perm: &[u8; N]) -> usize {
    let mut rank = 0;
    for i in 0..N {
        let mut smaller = 0;
        for j in i+1..N { if perm[j] < perm[i] { smaller += 1; } }
        rank += smaller * factorial(N - i - 1);
    }
    rank
}
fn factorial(n: usize) -> usize { (1..=n).product() }
fn binomial(n: usize, k: usize) -> usize {
    if k > n { return 0; }
    if k == 0 || k == n { return 1; }
    let k = k.min(n - k);
    let mut r = 1;
    for i in 1..=k { r = r * (n + 1 - i) / i; }
    r
}

/// Phase-1 move-index redundancy: same face, or opposite-face in non-canonical order.
/// Canonical: lower face index first (U<D, F<B, L<R) — eliminates commuting-pair duplicates.
fn is_redundant_p1(path: &[usize], next: usize) -> bool {
    if let Some(&last) = path.last() {
        let lf = PHASE1_MOVE_FACES[last] as usize;
        let nf = PHASE1_MOVE_FACES[next] as usize;
        if lf == nf { return true; }                       // same face
        if lf / 2 == nf / 2 && lf > nf { return true; }  // opposite, non-canonical
    }
    false
}

/// Phase-2 move-index redundancy (same rules, different face table).
fn is_redundant_p2(path: &[usize], next: usize) -> bool {
    if let Some(&last) = path.last() {
        let lf = PHASE2_MOVE_FACES[last] as usize;
        let nf = PHASE2_MOVE_FACES[next] as usize;
        if lf == nf { return true; }
        if lf / 2 == nf / 2 && lf > nf { return true; }
    }
    false
}

/// Legacy same-face check used only for scramble generation.
fn is_redundant(path: &[Move], next: Move) -> bool {
    if let Some(&last) = path.last() { (last as u8 / 3) == (next as u8 / 3) } else { false }
}

fn normalize_moves(moves: &[Move]) -> Vec<Move> {
    let mut out = Vec::with_capacity(moves.len());
    for &m in moves {
        if let Some(&last) = out.last() {
            if move_face(last) == move_face(m) {
                out.pop();
                let p = (move_power(last) + move_power(m)) % 4;
                if p != 0 { out.push(move_from_face_power(move_face(m), p)); }
                continue;
            }
        }
        out.push(m);
    }
    out
}

fn move_face(m: Move) -> usize { m as usize / 3 }
fn move_power(m: Move) -> usize {
    match m {
        Move::U|Move::D|Move::F|Move::B|Move::L|Move::R => 1,
        Move::U2|Move::D2|Move::F2|Move::B2|Move::L2|Move::R2 => 2,
        Move::U3|Move::D3|Move::F3|Move::B3|Move::L3|Move::R3 => 3,
    }
}
fn move_from_face_power(face: usize, power: usize) -> Move {
    match (face, power) {
        (0,1)=>Move::U, (0,2)=>Move::U2, (0,3)=>Move::U3,
        (1,1)=>Move::D, (1,2)=>Move::D2, (1,3)=>Move::D3,
        (2,1)=>Move::F, (2,2)=>Move::F2, (2,3)=>Move::F3,
        (3,1)=>Move::B, (3,2)=>Move::B2, (3,3)=>Move::B3,
        (4,1)=>Move::L, (4,2)=>Move::L2, (4,3)=>Move::L3,
        (5,1)=>Move::R, (5,2)=>Move::R2, (5,3)=>Move::R3,
        _ => unreachable!("invalid face or power"),
    }
}

fn should_stop_search(stats: &mut SolveStats, t0: Instant, limit: Option<Duration>) -> bool {
    if stats.timed_out { return true; }
    if let Some(lim) = limit {
        let nodes = stats.phase1_nodes + stats.phase2_nodes;
        if nodes.is_multiple_of(4096) && t0.elapsed() >= lim { stats.timed_out = true; return true; }
    }
    false
}

// ---------- 9. BASE_MOVES ----------
// Index map: Edges: UF,UR,UB,UL,DF,DR,DB,DL,FR,FL,BR,BL
//            Corners: URF,UFL,ULB,UBR,DFR,DFL,DBL,DBR

const BASE_MOVES: [CubeState; 6] = [
    // U
    CubeState { ep:[3,0,1,2,4,5,6,7,8,9,10,11], eo:[0;12], cp:[3,0,1,2,4,5,6,7], co:[0;8] },
    // D
    CubeState { ep:[0,1,2,3,5,6,7,4,8,9,10,11], eo:[0;12], cp:[0,1,2,3,5,6,7,4], co:[0;8] },
    // F
    CubeState {
        ep:[9,1,2,3,8,5,6,7,0,4,10,11], eo:[1,0,0,0,1,0,0,0,1,1,0,0],
        cp:[1,5,2,3,0,4,6,7], co:[1,2,0,0,2,1,0,0],
    },
    // B
    CubeState {
        ep:[0,1,10,3,4,5,11,7,8,9,6,2], eo:[0,0,1,0,0,0,1,0,0,0,1,1],
        cp:[0,1,3,7,4,5,2,6], co:[0,0,1,2,0,0,2,1],
    },
    // L
    CubeState {
        ep:[0,1,2,11,4,5,6,9,8,3,10,7], eo:[0;12],
        cp:[0,2,6,3,4,1,5,7], co:[0,1,2,0,0,2,1,0],
    },
    // R
    CubeState {
        ep:[0,8,2,3,4,10,6,7,5,9,1,11], eo:[0;12],
        cp:[4,1,2,0,7,5,6,3], co:[2,0,0,1,1,0,0,2],
    },
];

// ---------- 10. Stats, RNG, Scaffolding ----------

#[derive(Debug)]
struct BenchmarkStats {
    solves: usize, failures: usize, timeouts: usize,
    min_moves: usize, max_moves: usize, total_moves: usize,
    min_time: Duration, max_time: Duration, total_time: Duration,
}
impl BenchmarkStats {
    fn new() -> Self {
        Self { solves:0, failures:0, timeouts:0,
               min_moves:usize::MAX, max_moves:0, total_moves:0,
               min_time:Duration::MAX, max_time:Duration::ZERO, total_time:Duration::ZERO }
    }
    fn record_success(&mut self, moves: usize, t: Duration) {
        self.solves += 1;
        self.min_moves = self.min_moves.min(moves); self.max_moves = self.max_moves.max(moves);
        self.total_moves += moves;
        self.min_time = self.min_time.min(t); self.max_time = self.max_time.max(t);
        self.total_time += t;
    }
    fn record_failure(&mut self) { self.failures += 1; }
    fn record_timeout(&mut self) { self.timeouts += 1; }
    fn avg_moves(&self) -> f64 { self.total_moves as f64 / self.solves as f64 }
    fn avg_time(&self) -> Duration { Duration::from_secs_f64(self.total_time.as_secs_f64() / self.solves as f64) }
}

struct SimpleRng { state: u64 }
impl SimpleRng {
    fn new(seed: u64) -> Self { Self { state: seed } }
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.state
    }
    fn next_index(&mut self, len: usize) -> usize { (self.next_u64() as usize) % len }
}

fn main() {
    let mode = env::args().nth(1);
    println!("Initializing Solver and Tables...");
    let init_t = Instant::now();
    let solver = DominoSolver::new();
    println!("Solver initialization took {:.2?}.", init_t.elapsed());

    if mode.as_deref() == Some("competition") {
        run_benchmark("competition", &solver,
            COMPETITION_SOLVES, COMPETITION_SCRAMBLE_LEN, COMPETITION_SEED, COMPETITION_SOLVE_TIMEOUT_MS);
        return;
    }

    run_profiled_scramble("basic commutator", &solver, &[Move::R, Move::U, Move::R3, Move::U3]);
    run_profiled_scramble("mixed 6-move",    &solver, &[Move::F, Move::R, Move::U, Move::R3, Move::U3, Move::F3]);
    run_profiled_scramble("phase-2 heavy",   &solver, &[Move::U, Move::R2, Move::D, Move::L2, Move::U3, Move::F2]);
    run_benchmark("short", &solver,
        BENCHMARK_SOLVES, BENCHMARK_SCRAMBLE_LEN, BENCHMARK_SEED, BENCHMARK_SOLVE_TIMEOUT_MS);
}

fn run_profiled_scramble(label: &str, solver: &DominoSolver, scramble: &[Move]) {
    let mut cube = CubeState::new();
    for &m in scramble { cube.apply_move(m); }
    println!("\nSolving {}: {:?}", label, scramble);
    let (solution, stats) = solver.solve_profiled(&cube);
    if let Some(sol) = solution {
        println!("Solved in {:.2?}.", stats.total_elapsed);
        println!("Moves ({}): {:?}", sol.len(), sol);
        println!(
            "Profile: p1 depth {:?}, {} p1 sols, {} p1 nodes, {} p1 prunes, p1 {:.2?}; \
             {} p2 attempts, {} p2 nodes, p2 {:.2?}.",
            stats.phase1_depth, stats.phase1_solutions, stats.phase1_nodes,
            stats.phase1_pruned, stats.phase1_elapsed,
            stats.phase2_attempts, stats.phase2_nodes, stats.phase2_elapsed,
        );
        let mut check = cube;
        for &m in &sol { check.apply_move(m); }
        assert!(check.is_solved(), "Solution produced an invalid state!");
        println!("State verified: 100% Solved.");
    } else {
        println!("Failed. Profile: {:?}", stats);
    }
}

fn run_benchmark(label: &str, solver: &DominoSolver,
    solves: usize, scramble_len: usize, seed: u64, timeout_ms: u64)
{
    let mut rng = SimpleRng::new(seed);
    println!("\nRunning {} benchmark: {} scrambles, len {}, seed {:#x}, timeout {}ms, {} threads.",
        label, solves, scramble_len, seed, timeout_ms, rayon::current_num_threads());

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
        .map(|cube| solver.solve_profiled_with_time_limit(
            cube, Some(Duration::from_millis(timeout_ms))))
        .collect();

    // Aggregate results (sequential, order preserved by rayon's collect).
    let mut stats = BenchmarkStats::new();
    for (i, (sol, ss)) in results.iter().enumerate() {
        if let Some(sol) = sol {
            assert!(cubes[i].apply_moves(sol).is_solved(), "Solve {} invalid!", i+1);
            stats.record_success(sol.len(), ss.total_elapsed);
        } else if ss.timed_out { stats.record_timeout(); } else { stats.record_failure(); }

        if (i+1) % 10 == 0 || i+1 == solves {
            if stats.solves > 0 {
                println!("  completed {}/{} solves; avg {:.2} moves, avg {:.2?}.",
                    i+1, solves, stats.avg_moves(), stats.avg_time());
            } else {
                println!("  completed {}/{} solves; no successful solves yet.", i+1, solves);
            }
        }
    }

    println!("Benchmark completed in {:.2?} wall clock ({} threads).",
        bm_t.elapsed(), rayon::current_num_threads());
    println!("Successful solves: {}/{}.", stats.solves, solves);
    if stats.solves > 0 {
        println!("Moves: min {}, avg {:.2}, max {}.", stats.min_moves, stats.avg_moves(), stats.max_moves);
        println!("Time (per-solve): min {:.2?}, avg {:.2?}, max {:.2?}.",
            stats.min_time, stats.avg_time(), stats.max_time);
    }
    if stats.failures > 0 { println!("Failures: {}.", stats.failures); }
    if stats.timeouts > 0 { println!("Timeouts: {}.", stats.timeouts); }
}

fn random_scramble(rng: &mut SimpleRng, len: usize) -> Vec<Move> {
    let mut s = Vec::with_capacity(len);
    while s.len() < len {
        let next = PHASE1_MOVES[rng.next_index(PHASE1_MOVES.len())];
        if !is_redundant(&s, next) { s.push(next); }
    }
    s
}



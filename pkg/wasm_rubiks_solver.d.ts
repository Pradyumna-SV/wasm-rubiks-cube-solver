/* tslint:disable */
/* eslint-disable */

export class WasmSolver {
    free(): void;
    [Symbol.dispose](): void;
    /**
     * Create a new solver instance with embedded tables (compiled into the WASM binary).
     */
    static new_embedded(): WasmSolver;
    /**
     * Create a new solver instance by loading precomputed pruning tables from the provided byte slice.
     */
    constructor(data: Uint8Array);
    /**
     * Solve a scramble given as a space-separated string (e.g. "R U R' F2").
     * Returns the space-separated solution moves, or None/null if no solution is found or the scramble is invalid.
     */
    solve(scramble: string): string | undefined;
}

export type InitInput = RequestInfo | URL | Response | BufferSource | WebAssembly.Module;

export interface InitOutput {
    readonly memory: WebAssembly.Memory;
    readonly __wbg_wasmsolver_free: (a: number, b: number) => void;
    readonly wasmsolver_new_embedded: () => [number, number, number];
    readonly wasmsolver_new_with_bytes: (a: number, b: number) => [number, number, number];
    readonly wasmsolver_solve: (a: number, b: number, c: number) => [number, number];
    readonly __wbindgen_externrefs: WebAssembly.Table;
    readonly __externref_table_dealloc: (a: number) => void;
    readonly __wbindgen_malloc: (a: number, b: number) => number;
    readonly __wbindgen_realloc: (a: number, b: number, c: number, d: number) => number;
    readonly __wbindgen_free: (a: number, b: number, c: number) => void;
    readonly __wbindgen_start: () => void;
}

export type SyncInitInput = BufferSource | WebAssembly.Module;

/**
 * Instantiates the given `module`, which can either be bytes or
 * a precompiled `WebAssembly.Module`.
 *
 * @param {{ module: SyncInitInput }} module - Passing `SyncInitInput` directly is deprecated.
 *
 * @returns {InitOutput}
 */
export function initSync(module: { module: SyncInitInput } | SyncInitInput): InitOutput;

/**
 * If `module_or_path` is {RequestInfo} or {URL}, makes a request and
 * for everything else, calls `WebAssembly.instantiate` directly.
 *
 * @param {{ module_or_path: InitInput | Promise<InitInput> }} module_or_path - Passing `InitInput` directly is deprecated.
 *
 * @returns {Promise<InitOutput>}
 */
export default function __wbg_init (module_or_path?: { module_or_path: InitInput | Promise<InitInput> } | InitInput | Promise<InitInput>): Promise<InitOutput>;

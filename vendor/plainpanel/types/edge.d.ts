/**
 * The edge: where events over time become state.
 * Streams (websockets, timers, pointer events) live out here; each event
 * lands in the store as a batched signal write.
 */
import { type Computed } from './signals';
export interface Series {
    /** Append a sample; null marks a gap (a dead source draws a hole, not a frozen line). */
    push(v: number | null): void;
    clear(): void;
    /** Reactive read — an effect that calls read() re-runs on every push/clear. */
    read(): readonly (number | null)[];
    readonly capacity: number;
}
/**
 * Fixed-capacity rolling buffer for live metrics (sparklines, loss curves).
 * Mutates in place — no per-sample copying — and notifies through one
 * internal version signal, so consumers never call trigger() themselves.
 */
export declare function series(capacity?: number): Series;
export interface SocketOptions {
    /** Called with JSON.parse'd data, inside a batch — write signals freely. */
    onMessage: (data: unknown) => void;
    /** Reconnect delay after a drop. Default 1000ms. Infinite retries until close(). */
    reconnectMs?: number;
}
export interface Socket {
    readonly connected: Computed<boolean>;
    /** Sends JSON; returns false (and drops the message) when not connected. */
    send(data: unknown): boolean;
    close(): void;
}
/**
 * WebSocket → store, with auto-reconnect. `url` may be a function so each
 * (re)connect can build a cursor query like `?since=${lastSeq()}`.
 */
export declare function connect(url: string | (() => string), opts: SocketOptions): Socket;

/**
 * Reactive core: alien-signals, re-exported with one hardening rule.
 *
 * Signals are callables: read with `s()`, write with `s(next)`.
 * Every signal and computed is frozen at creation so the classic mistake
 * `s.value = x` throws a TypeError (modules are strict mode) instead of
 * silently assigning a dead property to a function object.
 */
import { isSignal, isComputed, trigger } from 'alien-signals';
/** Read with `s()`, write with `s(next)`. Writes of an identical value (===) are no-ops. */
export type Signal<T> = {
    (): T;
    (next: T): void;
};
/** Read-only derived value; lazily cached, recomputed only when a dependency changed. */
export type Computed<T> = () => T;
/** Anything an effect or binding can read reactively. */
export type Readable<T> = Signal<T> | Computed<T>;
/** Returned by effect/effectScope/bind/panel — call once to tear down. */
export type Stop = () => void;
export declare function signal<T>(initial: T): Signal<T>;
export declare function computed<T>(getter: (previous?: T) => T): Computed<T>;
/**
 * Runs `fn` now and again whenever any signal it read changes.
 * `fn` may return a cleanup function; it runs before each re-run and on stop.
 * Effects created inside another effect are cleaned up when the outer re-runs.
 * Non-function returns are discarded, so `effect(() => arr.push(x))` — an
 * arrow's implicit return — can't be mistaken for a cleanup and crash later.
 */
export declare function effect(fn: () => unknown): Stop;
/** Groups every effect created inside `fn`; the returned Stop disposes them all. */
export declare function effectScope(fn: () => void): Stop;
/** Apply several writes as one atomic update: effects run once, after all writes. */
export declare function batch(fn: () => void): void;
/** Read signals inside `fn` without the enclosing effect subscribing to them. */
export declare function untracked<T>(fn: () => T): T;
export { trigger, isSignal, isComputed };

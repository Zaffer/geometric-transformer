/**
 * Low-level element ⇄ signal bindings. One binding is either:
 *   - one effect writing one DOM property (signal → DOM), or
 *   - one event listener writing one signal (DOM → signal).
 * Both the attribute binder and the panel builder are built from these.
 */
import { type Readable, type Signal, type Stop } from './signals';
type ValueElement = HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement;
/** el.textContent tracks the source. */
export declare function bindText(el: Element, source: Readable<unknown>, format?: (v: unknown) => string): Stop;
/** el.hidden tracks !source — uses the native hidden attribute, no CSS involved. */
export declare function bindShow(el: HTMLElement, source: Readable<unknown>): Stop;
/** el.disabled tracks source. State disables controls; it never hides or moves them. */
export declare function bindDisabled(el: Element & {
    disabled: boolean;
}, source: Readable<unknown>): Stop;
/**
 * Two-way: form control value ⇄ signal.
 * The signal's current type decides coercion (number/boolean/string), so a
 * range slider bound to a number signal round-trips as a number, and a NaN
 * from a half-typed number input is never written into the store.
 * Programmatic writes don't fire 'input', and the signal's === short-circuit
 * kills the echo from our own writeback, so this cannot loop.
 *
 * Element-specific behavior:
 *   checkbox            checked ⇄ boolean signal
 *   radio               checked ⇄ (signal === this radio's value); one signal per group
 *   select[multiple]    selected options ⇄ string[] signal
 *   input[type=file]    one-way DOM → signal (browsers forbid setting a file
 *                       input's value); the signal receives File[]
 *   <details>           open ⇄ boolean signal (via the toggle event)
 *   everything else     value string ⇄ signal, coerced to the signal's type
 */
export declare function bindValue(el: ValueElement | HTMLDetailsElement, sig: Signal<any>): Stop;
/** One-way value display for <progress>/<meter> — no input events exist here. */
export declare function bindGauge(el: HTMLProgressElement | HTMLMeterElement, source: Readable<unknown>): Stop;
/** Whole-subtree disable via the native inert attribute: focus, clicks, and a11y. */
export declare function bindInert(el: HTMLElement, source: Readable<unknown>): Stop;
/** Mouse wheel nudges a range/number input by its step and writes the signal. */
export declare function bindWheel(el: HTMLInputElement, sig: Signal<number>): Stop;
/** addEventListener with a Stop, so listeners tear down with their scope. */
export declare function listen<K extends keyof HTMLElementEventMap>(el: EventTarget, type: K | string, handler: (e: Event) => void): Stop;
export {};

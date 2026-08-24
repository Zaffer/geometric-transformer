/**
 * Programmatic panel builder — the second entry point, for quick experiment
 * panels. Generates the same native elements you would write by hand
 * (details/summary, label, output, input) bound with the same primitives the
 * attribute binder uses. No CSS required; the optional plainpanel.css theme
 * targets .pp-panel for the polytopy look.
 *
 *   const p = panel('Training');
 *   p.slider('Learning rate', params.learningRate, { min: 0.001, max: 0.1, step: 0.001 });
 *   p.button(views.trainLabel, actions.train, { disabled: views.trainLocked });
 */
import { type Readable, type Signal } from './signals';
export interface PanelOptions {
    /** Where to append the panel. Default: document.body. */
    parent?: Element;
    /** Start expanded. Default: true. */
    open?: boolean;
}
export interface SliderOptions {
    min: number;
    max: number;
    step: number;
    /** Mouse wheel nudges by step. Default: true. */
    wheel?: boolean;
    /** Formats the value readout. Default: String. */
    format?: (v: number) => string;
}
export type SelectOption = string | number | {
    value: string | number;
    label: string;
};
export declare function panel(title: string, opts?: PanelOptions): Panel;
export declare class Panel {
    readonly el: HTMLDetailsElement;
    private readonly body;
    private readonly stops;
    private readonly folders;
    constructor(title: string, open: boolean);
    slider(label: string, sig: Signal<number>, opts: SliderOptions): this;
    number(label: string, sig: Signal<number>, opts?: Partial<Pick<SliderOptions, 'min' | 'max' | 'step'>>): this;
    text(label: string, sig: Signal<string>): this;
    /** Native color picker bound to a '#rrggbb' string signal. */
    color(label: string, sig: Signal<string>): this;
    toggle(label: string, sig: Signal<boolean>): this;
    select(label: string, sig: Signal<string | number>, options: SelectOption[]): this;
    /** Consecutive buttons flow onto one line — native inline layout. */
    button(label: string | Readable<string>, onClick: (e: Event) => void, opts?: {
        disabled?: Readable<unknown>;
    }): this;
    /** Read-only value display: label + <output>. */
    readout(label: string, source: Readable<unknown>, format?: (v: unknown) => string): this;
    /** Escape hatch: put any element (a canvas, a video tile) into the panel. */
    add(el: Element): this;
    /** Nested collapsible group. Disposed with its parent. */
    folder(title: string, open?: boolean): Panel;
    /** Stops every binding and listener, recursively, and removes the element. */
    dispose(): void;
    private row;
    private field;
}

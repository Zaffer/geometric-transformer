// A main-area view (one tab). Views subscribe to the store themselves and
// only do work while active.

export interface View {
  readonly id: string;
  readonly title: string;
  readonly el: HTMLElement;
  activate(): void;
  deactivate(): void;
  themeChanged?(): void;
}

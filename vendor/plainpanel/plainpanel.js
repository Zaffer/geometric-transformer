// node_modules/alien-signals/esm/system.mjs
function createReactiveSystem({ update, notify, unwatched }) {
  return {
    link: link2,
    unlink: unlink2,
    propagate: propagate2,
    checkDirty: checkDirty2,
    shallowPropagate: shallowPropagate2
  };
  function link2(dep, sub, version) {
    const prevDep = sub.depsTail;
    if (prevDep !== void 0 && prevDep.dep === dep) {
      return;
    }
    const nextDep = prevDep !== void 0 ? prevDep.nextDep : sub.deps;
    if (nextDep !== void 0 && nextDep.dep === dep) {
      nextDep.version = version;
      sub.depsTail = nextDep;
      return;
    }
    const prevSub = dep.subsTail;
    if (prevSub !== void 0 && prevSub.version === version && prevSub.sub === sub) {
      return;
    }
    const newLink = sub.depsTail = dep.subsTail = {
      version,
      dep,
      sub,
      prevDep,
      nextDep,
      prevSub,
      nextSub: void 0
    };
    if (nextDep !== void 0) {
      nextDep.prevDep = newLink;
    }
    if (prevDep !== void 0) {
      prevDep.nextDep = newLink;
    } else {
      sub.deps = newLink;
    }
    if (prevSub !== void 0) {
      prevSub.nextSub = newLink;
    } else {
      dep.subs = newLink;
    }
  }
  function unlink2(link3, sub = link3.sub) {
    const { dep, prevDep, nextDep, nextSub, prevSub } = link3;
    if (nextDep !== void 0) {
      nextDep.prevDep = prevDep;
    } else {
      sub.depsTail = prevDep;
    }
    if (prevDep !== void 0) {
      prevDep.nextDep = nextDep;
    } else {
      sub.deps = nextDep;
    }
    if (nextSub !== void 0) {
      nextSub.prevSub = prevSub;
    } else {
      dep.subsTail = prevSub;
    }
    if (prevSub !== void 0) {
      prevSub.nextSub = nextSub;
    } else if ((dep.subs = nextSub) === void 0) {
      unwatched(dep);
    }
    return nextDep;
  }
  function propagate2(link3, innerWrite) {
    let next = link3.nextSub;
    let stack;
    top: do {
      const sub = link3.sub;
      let flags = sub.flags;
      if (!(flags & (4 | 8 | 16 | 32))) {
        sub.flags = flags | 32;
        if (innerWrite) {
          sub.flags |= 8;
        }
      } else if (!(flags & (4 | 8))) {
        flags = 0;
      } else if (!(flags & 4)) {
        sub.flags = flags & ~8 | 32;
      } else if (!(flags & (16 | 32)) && isValidLink(link3, sub)) {
        sub.flags = flags | (8 | 32);
        flags &= 1;
      } else {
        flags = 0;
      }
      if (flags & 2) {
        notify(sub);
      }
      if (flags & 1) {
        const subSubs = sub.subs;
        if (subSubs !== void 0) {
          const nextSub = (link3 = subSubs).nextSub;
          if (nextSub !== void 0) {
            stack = { value: next, prev: stack };
            next = nextSub;
          }
          continue;
        }
      }
      if ((link3 = next) !== void 0) {
        next = link3.nextSub;
        continue;
      }
      while (stack !== void 0) {
        link3 = stack.value;
        stack = stack.prev;
        if (link3 !== void 0) {
          next = link3.nextSub;
          continue top;
        }
      }
      break;
    } while (true);
  }
  function checkDirty2(link3, sub) {
    let stack;
    let checkDepth = 0;
    let dirty = false;
    top: do {
      const dep = link3.dep;
      const flags = dep.flags;
      if (sub.flags & 16) {
        dirty = true;
      } else if ((flags & (1 | 16)) === (1 | 16)) {
        const subs = dep.subs;
        if (update(dep)) {
          if (subs.nextSub !== void 0) {
            shallowPropagate2(subs);
          }
          dirty = true;
        }
      } else if ((flags & (1 | 32)) === (1 | 32)) {
        stack = { value: link3, prev: stack };
        link3 = dep.deps;
        sub = dep;
        ++checkDepth;
        continue;
      }
      if (!dirty) {
        const nextDep = link3.nextDep;
        if (nextDep !== void 0) {
          link3 = nextDep;
          continue;
        }
      }
      while (checkDepth--) {
        link3 = stack.value;
        stack = stack.prev;
        if (dirty) {
          const subs = sub.subs;
          if (update(sub)) {
            if (subs.nextSub !== void 0) {
              shallowPropagate2(subs);
            }
            sub = link3.sub;
            continue;
          }
          dirty = false;
        } else {
          sub.flags &= ~32;
        }
        sub = link3.sub;
        const nextDep = link3.nextDep;
        if (nextDep !== void 0) {
          link3 = nextDep;
          continue top;
        }
      }
      return dirty && !!sub.flags;
    } while (true);
  }
  function shallowPropagate2(link3) {
    do {
      const sub = link3.sub;
      const flags = sub.flags;
      if ((flags & (32 | 16)) === 32) {
        sub.flags = flags | 16;
        if ((flags & (2 | 4)) === 2) {
          notify(sub);
        }
      }
    } while ((link3 = link3.nextSub) !== void 0);
  }
  function isValidLink(checkLink, sub) {
    let link3 = sub.depsTail;
    while (link3 !== void 0) {
      if (link3 === checkLink) {
        return true;
      }
      link3 = link3.prevDep;
    }
    return false;
  }
}

// node_modules/alien-signals/esm/index.mjs
var HasChildEffect = 64;
var cycle = 0;
var runDepth = 0;
var batchDepth = 0;
var notifyIndex = 0;
var queuedLength = 0;
var activeSub;
var queued = [];
var { link, unlink, propagate, checkDirty, shallowPropagate } = createReactiveSystem({
  update(node) {
    if ("getter" in node) {
      return updateComputed(node);
    }
    if ("currentValue" in node) {
      return updateSignal(node);
    }
    node.flags = 1;
    return true;
  },
  notify(effect3) {
    let insertIndex = queuedLength;
    let firstInsertedIndex = insertIndex;
    do {
      queued[insertIndex++] = effect3;
      effect3.flags &= ~2;
      effect3 = effect3.subs?.sub;
      if (effect3 === void 0 || !(effect3.flags & 2)) {
        break;
      }
    } while (true);
    queuedLength = insertIndex;
    while (firstInsertedIndex < --insertIndex) {
      const left = queued[firstInsertedIndex];
      queued[firstInsertedIndex++] = queued[insertIndex];
      queued[insertIndex] = left;
    }
  },
  unwatched(node) {
    if ("getter" in node) {
      if (node.depsTail !== void 0) {
        node.flags = 1 | 16;
        disposeAllDepsInReverse(node);
      }
    } else if ("currentValue" in node) {
    } else if ("fn" in node) {
      effectOper.call(node);
    } else {
      effectScopeOper.call(node);
    }
  }
});
function setActiveSub(sub) {
  const prevSub = activeSub;
  activeSub = sub;
  return prevSub;
}
function startBatch() {
  ++batchDepth;
}
function endBatch() {
  if (!--batchDepth) {
    flush();
  }
}
function isSignal(fn) {
  return fn.name === "bound " + signalOper.name;
}
function isComputed(fn) {
  return fn.name === "bound " + computedOper.name;
}
function signal(initialValue) {
  return signalOper.bind({
    currentValue: initialValue,
    pendingValue: initialValue,
    subs: void 0,
    subsTail: void 0,
    flags: 1
  });
}
function computed(getter) {
  return computedOper.bind({
    value: void 0,
    subs: void 0,
    subsTail: void 0,
    deps: void 0,
    depsTail: void 0,
    flags: 0,
    getter
  });
}
function effect(fn) {
  const e = {
    fn,
    cleanup: void 0,
    subs: void 0,
    subsTail: void 0,
    deps: void 0,
    depsTail: void 0,
    flags: 2 | 4
  };
  const prevSub = setActiveSub(e);
  if (prevSub !== void 0) {
    link(e, prevSub, 0);
    prevSub.flags |= HasChildEffect;
  }
  try {
    ++runDepth;
    e.cleanup = e.fn();
  } finally {
    --runDepth;
    activeSub = prevSub;
    e.flags &= ~4;
  }
  return effectOper.bind(e);
}
function effectScope(fn) {
  const e = {
    deps: void 0,
    depsTail: void 0,
    subs: void 0,
    subsTail: void 0,
    flags: 1
  };
  const prevSub = setActiveSub(e);
  if (prevSub !== void 0) {
    link(e, prevSub, 0);
    prevSub.flags |= HasChildEffect;
  }
  try {
    fn();
  } finally {
    activeSub = prevSub;
  }
  return effectScopeOper.bind(e);
}
function trigger(fn) {
  const sub = {
    deps: void 0,
    depsTail: void 0,
    flags: 2
  };
  const prevSub = setActiveSub(sub);
  try {
    fn();
  } finally {
    activeSub = prevSub;
    sub.flags = 0;
    let link2 = sub.deps;
    while (link2 !== void 0) {
      const dep = link2.dep;
      link2 = unlink(link2, sub);
      const subs = dep.subs;
      if (subs !== void 0) {
        propagate(subs, !!runDepth);
        shallowPropagate(subs);
      }
    }
    if (!batchDepth) {
      flush();
    }
  }
}
function updateComputed(c) {
  if (c.flags & HasChildEffect) {
    let link2 = c.depsTail;
    while (link2 !== void 0) {
      const prev = link2.prevDep;
      const dep = link2.dep;
      if (!("getter" in dep) && !("currentValue" in dep)) {
        unlink(link2, c);
      }
      link2 = prev;
    }
  }
  c.depsTail = void 0;
  c.flags = 1 | 4;
  const prevSub = setActiveSub(c);
  try {
    ++cycle;
    const oldValue = c.value;
    return oldValue !== (c.value = c.getter(oldValue));
  } finally {
    activeSub = prevSub;
    c.flags &= ~4;
    purgeDeps(c);
  }
}
function updateSignal(s) {
  s.flags = 1;
  return s.currentValue !== (s.currentValue = s.pendingValue);
}
function run(e) {
  const flags = e.flags;
  if (flags & 16 || flags & 32 && checkDirty(e.deps, e)) {
    if (flags & HasChildEffect) {
      let link2 = e.depsTail;
      while (link2 !== void 0) {
        const prev = link2.prevDep;
        const dep = link2.dep;
        if (!("getter" in dep) && !("currentValue" in dep)) {
          unlink(link2, e);
        }
        link2 = prev;
      }
    }
    if (e.cleanup) {
      runCleanup(e);
      if (!e.flags) {
        return;
      }
    }
    e.depsTail = void 0;
    e.flags = 2 | 4;
    const prevSub = setActiveSub(e);
    try {
      ++cycle;
      ++runDepth;
      e.cleanup = e.fn();
    } finally {
      --runDepth;
      activeSub = prevSub;
      e.flags &= ~4;
      purgeDeps(e);
    }
  } else if (e.deps !== void 0) {
    e.flags = 2 | flags & HasChildEffect;
  }
}
function flush() {
  try {
    while (notifyIndex < queuedLength) {
      const effect3 = queued[notifyIndex];
      queued[notifyIndex++] = void 0;
      run(effect3);
    }
  } finally {
    while (notifyIndex < queuedLength) {
      const effect3 = queued[notifyIndex];
      queued[notifyIndex++] = void 0;
      effect3.flags |= 2 | 8;
    }
    notifyIndex = 0;
    queuedLength = 0;
  }
}
function computedOper() {
  const flags = this.flags;
  if (flags & 16 || flags & 32 && (checkDirty(this.deps, this) || (this.flags = flags & ~32, false))) {
    if (updateComputed(this)) {
      const subs = this.subs;
      if (subs !== void 0) {
        shallowPropagate(subs);
      }
    }
  } else if (!flags) {
    this.flags = 1 | 4;
    const prevSub = setActiveSub(this);
    try {
      this.value = this.getter();
    } finally {
      activeSub = prevSub;
      this.flags &= ~4;
    }
  }
  const sub = activeSub;
  if (sub !== void 0) {
    link(this, sub, cycle);
  }
  return this.value;
}
function signalOper(...value) {
  if (value.length) {
    if (this.pendingValue !== (this.pendingValue = value[0])) {
      this.flags = 1 | 16;
      const subs = this.subs;
      if (subs !== void 0) {
        propagate(subs, !!runDepth);
        if (!batchDepth) {
          flush();
        }
      }
    }
  } else {
    if (this.flags & 16) {
      if (updateSignal(this)) {
        const subs = this.subs;
        if (subs !== void 0) {
          shallowPropagate(subs);
        }
      }
    }
    const sub = activeSub;
    if (sub !== void 0) {
      link(this, sub, cycle);
    }
    return this.currentValue;
  }
}
function runCleanup(e) {
  const cleanup = e.cleanup;
  e.cleanup = void 0;
  const prevSub = activeSub;
  activeSub = void 0;
  try {
    cleanup();
  } finally {
    activeSub = prevSub;
  }
}
function effectOper() {
  effectScopeOper.call(this);
  if (this.cleanup) {
    runCleanup(this);
  }
}
function effectScopeOper() {
  this.flags = 0;
  disposeAllDepsInReverse(this);
  const sub = this.subs;
  if (sub !== void 0) {
    unlink(sub);
  }
}
function disposeAllDepsInReverse(sub) {
  let link2 = sub.depsTail;
  while (link2 !== void 0) {
    const prev = link2.prevDep;
    unlink(link2, sub);
    link2 = prev;
  }
}
function purgeDeps(sub) {
  const depsTail = sub.depsTail;
  let dep = depsTail !== void 0 ? depsTail.nextDep : sub.deps;
  while (dep !== void 0) {
    dep = unlink(dep, sub);
  }
}

// src/signals.ts
function signal2(initial) {
  return Object.freeze(signal(initial));
}
function computed2(getter) {
  return Object.freeze(computed(getter));
}
function effect2(fn) {
  return effect(() => {
    const cleanup = fn();
    return typeof cleanup === "function" ? cleanup : void 0;
  });
}
function effectScope2(fn) {
  return effectScope(fn);
}
function batch(fn) {
  startBatch();
  try {
    fn();
  } finally {
    endBatch();
  }
}
function untracked(fn) {
  const previous = setActiveSub(void 0);
  try {
    return fn();
  } finally {
    setActiveSub(previous);
  }
}

// src/bindings.ts
function bindText(el, source, format) {
  return effect2(() => {
    el.textContent = format ? format(source()) : String(source());
  });
}
function bindShow(el, source) {
  return effect2(() => {
    el.hidden = !source();
  });
}
function bindDisabled(el, source) {
  return effect2(() => {
    el.disabled = !!source();
  });
}
function bindValue(el, sig) {
  if (el instanceof HTMLDetailsElement) {
    const stop2 = effect2(() => {
      el.open = !!sig();
    });
    const onToggle = () => sig(el.open);
    el.addEventListener("toggle", onToggle);
    return () => {
      stop2();
      el.removeEventListener("toggle", onToggle);
    };
  }
  const type = el.type;
  if (type === "file") {
    const onInput2 = () => sig([...el.files ?? []]);
    el.addEventListener("input", onInput2);
    return () => el.removeEventListener("input", onInput2);
  }
  const kind = untracked(() => typeof sig());
  const coerce = (raw) => kind === "number" ? Number(raw) : raw;
  let stop;
  let onInput;
  if (type === "checkbox") {
    stop = effect2(() => {
      el.checked = !!sig();
    });
    onInput = () => sig(el.checked);
  } else if (type === "radio") {
    stop = effect2(() => {
      el.checked = sig() === coerce(el.value);
    });
    onInput = () => {
      if (el.checked) sig(coerce(el.value));
    };
  } else if (el instanceof HTMLSelectElement && el.multiple) {
    stop = effect2(() => {
      const selected = sig();
      if (!Array.isArray(selected)) {
        throw new Error("plainpanel: a select[multiple] binding needs a signal holding an array");
      }
      for (const option of el.options) option.selected = selected.includes(option.value);
    });
    onInput = () => sig([...el.selectedOptions].map((o) => o.value));
  } else {
    stop = effect2(() => {
      el.value = String(sig());
    });
    onInput = () => {
      if (kind === "number") {
        const n = Number(el.value);
        if (!Number.isNaN(n)) sig(n);
        return;
      }
      sig(el.value);
    };
  }
  el.addEventListener("input", onInput);
  return () => {
    stop();
    el.removeEventListener("input", onInput);
  };
}
function bindGauge(el, source) {
  return effect2(() => {
    el.value = Number(source()) || 0;
  });
}
function bindInert(el, source) {
  return effect2(() => {
    el.inert = !!source();
  });
}
function bindWheel(el, sig) {
  const onWheel = (e) => {
    e.preventDefault();
    if (e.deltaY < 0) el.stepUp();
    else el.stepDown();
    sig(el.valueAsNumber);
  };
  el.addEventListener("wheel", onWheel, { passive: false });
  return () => el.removeEventListener("wheel", onWheel);
}
function listen(el, type, handler) {
  el.addEventListener(type, handler);
  return () => el.removeEventListener(type, handler);
}

// src/binder.ts
function isReadable(v) {
  return typeof v === "function" && (isSignal(v) || isComputed(v));
}
function walk(scope, path) {
  let current = scope;
  for (const key of path.split(".")) {
    if (isReadable(current)) current = current();
    if (current == null || !(key in Object(current))) {
      throw new Error(`plainpanel: path "${path}" not found in scope (stopped at "${key}")`);
    }
    current = current[key];
  }
  return current;
}
function resolveValue(scope, path) {
  const v = walk(scope, path);
  if (isReadable(v)) return v();
  if (typeof v === "function") {
    throw new Error(`plainpanel: path "${path}" resolves to a plain function \u2014 bindable values must be signal(), computed(), or plain data`);
  }
  return v;
}
function resolveTarget(scope, path) {
  return walk(scope, path);
}
var SELECTOR = "[data-text],[data-bind],[data-show],[data-disabled],[data-inert],[data-on],[data-each]";
function bind(root, scope) {
  const stops = [];
  const read = (path) => {
    untracked(() => resolveValue(scope, path));
    return () => resolveValue(scope, path);
  };
  const writable = (path, attr) => {
    const leaf = untracked(() => resolveTarget(scope, path));
    if (!(typeof leaf === "function" && isSignal(leaf))) {
      throw new Error(`plainpanel: ${attr}="${path}" must point to a signal(), got ${typeof leaf}`);
    }
    return ((...args) => {
      const target = resolveTarget(scope, path);
      return args.length ? target(args[0]) : target();
    });
  };
  const handler = (path) => {
    const leaf = untracked(() => resolveTarget(scope, path));
    if (typeof leaf !== "function" || isReadable(leaf)) {
      throw new Error(`plainpanel: data-on handler "${path}" is not a function`);
    }
    return (e) => resolveTarget(scope, path)(e);
  };
  const targets = [];
  if (root instanceof Element && root.matches(SELECTOR)) targets.push(root);
  targets.push(...root.querySelectorAll(SELECTOR));
  for (const el of targets) {
    const d = el.dataset;
    if (d.each !== void 0) {
      if (!(el instanceof HTMLTemplateElement)) {
        throw new Error(`plainpanel: data-each="${d.each}" only works on <template> elements`);
      }
      stops.push(bindEach(el, scope));
      continue;
    }
    if (d.text !== void 0) stops.push(bindText(el, read(d.text)));
    if (d.show !== void 0) stops.push(bindShow(el, read(d.show)));
    if (d.disabled !== void 0) {
      stops.push(bindDisabled(el, read(d.disabled)));
    }
    if (d.inert !== void 0) stops.push(bindInert(el, read(d.inert)));
    if (d.bind !== void 0) {
      if (el instanceof HTMLProgressElement || el instanceof HTMLMeterElement) {
        stops.push(bindGauge(el, read(d.bind)));
      } else {
        const sig = writable(d.bind, "data-bind");
        stops.push(bindValue(el, sig));
        if (d.wheel !== void 0) stops.push(bindWheel(el, sig));
      }
    }
    if (d.on !== void 0) {
      for (const pair of d.on.trim().split(/\s+/)) {
        const i = pair.indexOf(":");
        if (i < 1) throw new Error(`plainpanel: data-on="${pair}" must be "event:path"`);
        stops.push(listen(el, pair.slice(0, i), handler(pair.slice(i + 1))));
      }
    }
  }
  return () => {
    for (const stop of stops.splice(0)) stop();
  };
}
function bindEach(tpl, scope) {
  const path = (tpl.dataset.each ?? "").trim();
  const keyPath = (tpl.dataset.key ?? "").trim();
  if (!keyPath) {
    throw new Error(
      `plainpanel: data-each="${path}" requires data-key \u2014 a unique item field like data-key="id", data-key="$item" for primitive items, or data-key="$index" for explicitly positional rows`
    );
  }
  const keyOf = (item, index) => {
    if (keyPath === "$index") return index;
    let key = keyPath === "$item" ? item : walk(item, keyPath);
    if (isReadable(key)) key = key();
    if (typeof key === "object" && key !== null) {
      throw new Error(
        `plainpanel: data-key="${keyPath}" produced an object \u2014 keys must be primitive (fresh objects would defeat tracking); key by a field instead`
      );
    }
    return key;
  };
  let rows = /* @__PURE__ */ new Map();
  const removeRow = (row) => {
    row.stop();
    for (const node of row.nodes) node.remove();
  };
  const stopEffect = effect2(() => {
    const items = resolveValue(scope, path);
    if (!Array.isArray(items)) {
      throw new Error(`plainpanel: data-each="${path}" must read an array, got ${typeof items}`);
    }
    untracked(() => {
      const next = /* @__PURE__ */ new Map();
      let anchor = tpl;
      items.forEach((itemValue, i) => {
        const key = keyOf(itemValue, i);
        if (next.has(key)) {
          throw new Error(`plainpanel: duplicate data-key value "${String(key)}" in data-each="${path}"`);
        }
        let row = rows.get(key);
        if (row) {
          rows.delete(key);
          row.item(itemValue);
          row.index(i);
          if (anchor.nextSibling !== row.nodes[0]) {
            let ref = anchor;
            for (const node of row.nodes) {
              ref.after(node);
              ref = node;
            }
          }
        } else {
          const item = signal2(itemValue);
          const index = signal2(i);
          const clone = tpl.content.cloneNode(true);
          const rowScope = Object.assign(Object.create(scope), { $item: item, $index: index });
          const stop = bind(clone, rowScope);
          const nodes = [...clone.childNodes];
          anchor.after(clone);
          row = { item, index, stop, nodes };
        }
        next.set(key, row);
        anchor = row.nodes[row.nodes.length - 1] ?? anchor;
      });
      for (const stale of rows.values()) removeRow(stale);
      rows = next;
    });
  });
  return () => {
    stopEffect();
    for (const row of rows.values()) removeRow(row);
    rows.clear();
  };
}

// src/panel.ts
function panel(title, opts = {}) {
  const p = new Panel(title, opts.open ?? true);
  (opts.parent ?? document.body).appendChild(p.el);
  return p;
}
var Panel = class _Panel {
  el;
  body;
  stops = [];
  folders = [];
  constructor(title, open) {
    this.el = document.createElement("details");
    this.el.className = "pp-panel";
    this.el.open = open;
    const summary = document.createElement("summary");
    summary.textContent = title;
    this.body = document.createElement("div");
    this.body.className = "pp-body";
    this.el.append(summary, this.body);
  }
  slider(label, sig, opts) {
    const { row, labelEl } = this.row(label);
    const output = document.createElement("output");
    const input = document.createElement("input");
    input.type = "range";
    input.min = String(opts.min);
    input.max = String(opts.max);
    input.step = String(opts.step);
    labelEl.append(" ", output, document.createElement("br"), input);
    this.body.appendChild(row);
    const format = opts.format ?? String;
    this.stops.push(
      bindText(output, sig, (v) => format(v)),
      bindValue(input, sig)
    );
    if (opts.wheel ?? true) this.stops.push(bindWheel(input, sig));
    return this;
  }
  number(label, sig, opts = {}) {
    const input = document.createElement("input");
    input.type = "number";
    if (opts.min !== void 0) input.min = String(opts.min);
    if (opts.max !== void 0) input.max = String(opts.max);
    if (opts.step !== void 0) input.step = String(opts.step);
    return this.field(label, input, sig);
  }
  text(label, sig) {
    const input = document.createElement("input");
    input.type = "text";
    return this.field(label, input, sig);
  }
  /** Native color picker bound to a '#rrggbb' string signal. */
  color(label, sig) {
    const input = document.createElement("input");
    input.type = "color";
    return this.field(label, input, sig);
  }
  toggle(label, sig) {
    const { row, labelEl } = this.row("");
    const input = document.createElement("input");
    input.type = "checkbox";
    labelEl.append(input, ` ${label}`);
    this.body.appendChild(row);
    this.stops.push(bindValue(input, sig));
    return this;
  }
  select(label, sig, options) {
    const select = document.createElement("select");
    for (const opt of options) {
      const o = document.createElement("option");
      if (typeof opt === "object") {
        o.value = String(opt.value);
        o.textContent = opt.label;
      } else {
        o.value = String(opt);
        o.textContent = String(opt);
      }
      select.appendChild(o);
    }
    return this.field(label, select, sig);
  }
  /** Consecutive buttons flow onto one line — native inline layout. */
  button(label, onClick, opts = {}) {
    const button = document.createElement("button");
    if (typeof label === "string") button.textContent = label;
    else this.stops.push(bindText(button, label));
    this.stops.push(listen(button, "click", onClick));
    if (opts.disabled) this.stops.push(bindDisabled(button, opts.disabled));
    this.body.appendChild(button);
    return this;
  }
  /** Read-only value display: label + <output>. */
  readout(label, source, format) {
    const { row, labelEl } = this.row(label);
    const output = document.createElement("output");
    labelEl.append(" ", output);
    this.body.appendChild(row);
    this.stops.push(bindText(output, source, format));
    return this;
  }
  /** Escape hatch: put any element (a canvas, a video tile) into the panel. */
  add(el) {
    this.body.appendChild(el);
    return this;
  }
  /** Nested collapsible group. Disposed with its parent. */
  folder(title, open = true) {
    const child = new _Panel(title, open);
    child.el.classList.replace("pp-panel", "pp-folder");
    this.body.appendChild(child.el);
    this.folders.push(child);
    return child;
  }
  /** Stops every binding and listener, recursively, and removes the element. */
  dispose() {
    for (const folder of this.folders.splice(0)) folder.dispose();
    for (const stop of this.stops.splice(0)) stop();
    this.el.remove();
  }
  row(label) {
    const row = document.createElement("div");
    row.className = "pp-row";
    const labelEl = document.createElement("label");
    if (label) labelEl.append(label);
    row.appendChild(labelEl);
    return { row, labelEl };
  }
  field(label, input, sig) {
    const { row, labelEl } = this.row(label);
    labelEl.append(" ", input);
    this.body.appendChild(row);
    this.stops.push(bindValue(input, sig));
    return this;
  }
};

// src/edge.ts
function series(capacity = 600) {
  const buffer = [];
  const version = signal2(0);
  return Object.freeze({
    capacity,
    push(v) {
      buffer.push(v);
      if (buffer.length > capacity) buffer.shift();
      version(version() + 1);
    },
    clear() {
      buffer.length = 0;
      version(version() + 1);
    },
    read() {
      version();
      return buffer;
    }
  });
}
function connect(url, opts) {
  const reconnectMs = opts.reconnectMs ?? 1e3;
  const isConnected = signal2(false);
  let ws = null;
  let closed = false;
  let timer;
  const open = () => {
    try {
      ws = new WebSocket(typeof url === "function" ? url() : url);
    } catch (err) {
      console.error("plainpanel: connect() failed \u2014", err);
      return;
    }
    ws.onopen = () => isConnected(true);
    ws.onmessage = (e) => batch(() => opts.onMessage(JSON.parse(e.data)));
    ws.onclose = () => {
      isConnected(false);
      if (!closed) timer = setTimeout(open, reconnectMs);
    };
    ws.onerror = () => ws?.close();
  };
  open();
  return {
    connected: computed2(() => isConnected()),
    send(data) {
      if (ws?.readyState !== WebSocket.OPEN) return false;
      ws.send(JSON.stringify(data));
      return true;
    },
    close() {
      closed = true;
      clearTimeout(timer);
      ws?.close();
    }
  };
}
export {
  Panel,
  batch,
  bind,
  bindDisabled,
  bindGauge,
  bindInert,
  bindShow,
  bindText,
  bindValue,
  bindWheel,
  computed2 as computed,
  connect,
  effect2 as effect,
  effectScope2 as effectScope,
  isComputed,
  isSignal,
  listen,
  panel,
  resolveTarget,
  resolveValue,
  series,
  signal2 as signal,
  trigger,
  untracked
};

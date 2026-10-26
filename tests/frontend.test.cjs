const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const test = require("node:test");
const vm = require("node:vm");

const source = fs.readFileSync(path.join(__dirname, "../app/index.html"), "utf8")
  .match(/<script>([\s\S]*?)<\/script>/)[1];

function load(names, globals = {}) {
  const context = vm.createContext({ URLSearchParams, TextDecoder, clearTimeout, ...globals });
  for (const name of names) {
    const definition = source.match(new RegExp(`(?:async )?function ${name}\\([^]*?\\n\\}`))
      || source.match(new RegExp(`^const ${name} = .*;`, "m"));
    assert.ok(definition, `Missing function ${name}`);
    vm.runInContext(definition[0], context);
  }
  return context;
}

const deferred = () => {
  let resolve, reject;
  const promise = new Promise((yes, no) => { resolve = yes; reject = no; });
  return { promise, resolve, reject };
};

test("frontend script parses", () => { new vm.Script(source); });

test("stream reports HTTP failures and cancels a reader after a callback fails", async () => {
  const context = load(["sse"], { fetch: async () => ({ ok: false, status: 503, text: async () => "Unavailable" }) });
  await assert.rejects(context.sse("/stream", {}, () => {}), /503.*Unavailable/);
  let cancelled = false, released = false;
  context.fetch = async () => ({ ok: true, body: { getReader: () => ({
    read: async () => ({ value: new TextEncoder().encode('data: {"delta":"text"}\n\n'), done: false }),
    cancel: async () => { cancelled = true; },
    releaseLock: () => { released = true; },
  }) } });
  await assert.rejects(context.sse("/stream", {}, () => { throw new Error("Callback failed"); }), /Callback failed/);
  assert.ok(cancelled && released);
});

test("search results cannot overwrite a different model or Find mode", async () => {
  for (const change of [(state) => { state.model = "small"; }, (state) => { state.fmode = "text"; }]) {
    const request = deferred();
    const context = load(["words"], {
      state: { model: "flash", fmode: "words", q: "secret" }, findVersion: 0, qTimer: null,
      ID_RE: /^L(\d+)N(\d+)$/, found: null, get: () => request.promise,
      renderResults: () => assert.fail("Stale results were rendered"),
    });
    const pending = context.words();
    change(context.state);
    request.resolve({ results: [{ layer: 4, neuron: 1 }] });
    await pending;
    assert.equal(context.found, null);
  }
});

test("an older search cannot overwrite a newer search for the same query", async () => {
  const requests = [deferred(), deferred()]; let next = 0;
  const context = load(["words"], {
    state: { model: "flash", fmode: "words", q: "secret" }, findVersion: 0, qTimer: null,
    ID_RE: /^L(\d+)N(\d+)$/, found: null, get: () => requests[next++].promise, renderResults: () => {},
  });
  const old = context.words(), recent = context.words();
  requests[1].resolve({ results: ["recent"] }); await recent;
  requests[0].resolve({ results: ["old"] }); await old;
  assert.deepEqual(context.found.rows, ["recent"]);
});

test("a completed text search cannot restore results after switching Find modes", async () => {
  const request = deferred();
  const context = load(["runFind"], {
    state: { model: "flash", fmode: "text" }, job: null, findVersion: 0, found: null,
    setJob: (job) => { context.job = job; }, findBusy: () => {}, post: () => request.promise,
    renderResults: () => assert.fail("Stale results were rendered"),
  });
  const pending = context.runFind("text", "/fires", {}, "Reading");
  context.state.fmode = "contrast";
  request.resolve({ overall: ["old"] }); await pending;
  assert.equal(context.job, null);
  assert.equal(context.found, null);
});

test("model changes and column edits cannot mutate an active chat", async () => {
  class Element {
    constructor(cls = "") { this.className = cls; this.children = []; this.nodes = {}; this.handlers = {}; }
    append(child) { this.children.push(child); }
    querySelector(selector) { return this.nodes[selector] ||= new Element(); }
    addEventListener(name, callback) { this.handlers[name] = callback; }
  }
  const nodes = { cols: new Element(), csuggest: new Element() }, factors = [0, 1];
  const context = load(["setModel", "renderCols"], {
    job: "chat", state: { model: "flash" }, factors: () => factors, convs: [["first"], ["second"]],
    $: (name) => nodes[name], el: (tag, cls) => new Element(cls),
    ICON: {}, fcls: () => "", fglyph: String, paintCol: () => {}, CHAT_SUGGEST: [], toast: () => {},
  });
  await context.setModel("small");
  context.renderCols();
  nodes.cols.children[0].querySelector(".mini").handlers.click();
  nodes.cols.children.at(-1).querySelector("button").onclick();
  nodes.cols.children[0].querySelector(".glyph").onclick();
  assert.equal(context.state.model, "flash");
  assert.deepEqual(factors, [0, 1]);
  assert.deepEqual(context.convs, [["first"], ["second"]]);
});

test("agent setup failures release the busy state and preserve the prompt", async () => {
  const input = { value: "Fix this file" }, errors = [];
  const context = load(["sendAgent"], {
    job: null, session: null, laneBusy: {}, $: () => input,
    setJob: (job) => { context.job = job; },
    ensureSession: async () => { throw new Error("Network failed"); }, toast: (error) => errors.push(error),
  });
  await context.sendAgent();
  assert.equal(context.job, null);
  assert.equal(input.value, "Fix this file");
  assert.match(errors[0], /Network failed/);
});

test("model switching waits for warmup before permitting the next operation", async () => {
  const request = deferred();
  const context = load(["setModel"], {
    job: null, state: { model: "flash" }, setJob: (job) => { context.job = job; },
    save: () => {}, paintModel: () => {}, renderStack: () => {}, renderCols: () => {},
    renderLanes: () => {}, renderInspector: () => {}, paintSend: () => {}, poll: () => {},
    post: (url) => { assert.equal(url, "/api/warm"); return request.promise; },
  });
  const pending = context.setModel("flash");
  assert.equal(context.job, "loading");
  request.resolve({});
  assert.equal(await pending, true);
  assert.equal(context.job, null);
});

test("reset waits for sandbox closure and preserves state when closure fails", async () => {
  for (const failed of [false, true]) {
    const request = deferred(), session = { lanes: ["steered"] }, viewer = { hidden: false };
    const context = load(["dropSession"], {
      job: null, session, lanesEv: {}, laneFiles: {}, touched: {}, viewing: "file.py",
      setJob: (job) => { context.job = job; }, $: () => viewer,
      post: (url) => { assert.equal(url, "/api/agent/close"); return request.promise; }, renderLanes: () => {}, toast: () => {},
    });
    const pending = context.dropSession();
    assert.equal(context.session, session);
    assert.equal(context.job, "session");
    request.resolve(failed ? { error: "Still running" } : {});
    assert.equal(await pending, !failed);
    assert.equal(context.session, failed ? session : null);
    assert.equal(context.job, null);
    assert.equal(viewer.hidden, !failed);
  }
});

test("lens queues snapshot the neuron address and preserve routed expert zero", async () => {
  let url;
  const context = load(["expertKey", "lens", "pump"], {
    job: null, state: { model: "flash" }, lensCache: {}, lensQueue: [], lensBusy: 3,
    get: async (value) => { url = value; return { up: ["token"] }; },
  });
  const neuron = { layer: 3, neuron: 4, expert: 0 }, pending = context.lens(neuron);
  neuron.neuron = 8;
  context.lensBusy = 0; context.pump(); await pending;
  const query = new URLSearchParams(url.split("?")[1]);
  assert.equal(query.get("neuron"), "4");
  assert.equal(query.get("expert"), "0");
});

test("temporary lens failures can be retried", async () => {
  let count = 0;
  const context = load(["expertKey", "lens", "pump"], {
    job: null, state: { model: "flash" }, lensCache: {}, lensQueue: [], lensBusy: 0,
    get: async () => ++count === 1 ? { error: "Busy" } : { up: ["token"] },
  });
  const neuron = { layer: 3, neuron: 4, expert: 0 };
  await context.lens(neuron);
  assert.deepEqual((await context.lens(neuron)).up, ["token"]);
  assert.equal(count, 2);
});

/* ==========================================================
   NYC Room-Type Predictor: front-end logic
   Talks to the FastAPI endpoint: POST {API}/predict
   ========================================================== */
"use strict";

/* ---------- Config ---------- */

const DEFAULT_API = "http://127.0.0.1:8000";
const STORE_API = "nyc-room-api-url";
const STORE_HISTORY = "nyc-room-history";
const HISTORY_MAX = 6;

// scikit-learn sorts class labels alphabetically, so predict_proba()
// follows this order. If your API ever returns a "Classes" list, that wins.
const CLASS_ORDER = ["Entire home/apt", "Private room", "Shared room"];
const CLASS_META = {
  "Entire home/apt": { label: "Entire home or apartment", short: "E", color: "var(--home)" },
  "Private room":    { label: "Private room",             short: "P", color: "var(--private)" },
  "Shared room":     { label: "Shared room",              short: "S", color: "var(--shared)" },
};

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

/* ---------- Neighbourhoods the model knows, grouped by borough ---------- */

const HOODS = {
  "Bronx": ["Allerton","Baychester","Belmont","Bronxdale","Castle Hill","City Island","Claremont Village","Clason Point","Co-op City","Concourse","Concourse Village","East Morrisania","Eastchester","Edenwald","Fieldston","Fordham","Highbridge","Hunts Point","Kingsbridge","Longwood","Melrose","Morris Heights","Morris Park","Morrisania","Mott Haven","Mount Eden","Mount Hope","North Riverdale","Norwood","Olinville","Parkchester","Pelham Bay","Pelham Gardens","Port Morris","Riverdale","Schuylerville","Soundview","Spuyten Duyvil","Throgs Neck","Tremont","Unionport","University Heights","Van Nest","Wakefield","West Farms","Westchester Square","Williamsbridge","Woodlawn"],
  "Brooklyn": ["Bath Beach","Bay Ridge","Bedford-Stuyvesant","Bensonhurst","Bergen Beach","Boerum Hill","Borough Park","Brighton Beach","Brooklyn Heights","Brownsville","Bushwick","Canarsie","Carroll Gardens","Clinton Hill","Cobble Hill","Columbia St","Coney Island","Crown Heights","Cypress Hills","DUMBO","Downtown Brooklyn","Dyker Heights","East Flatbush","East New York","Flatbush","Flatlands","Fort Greene","Fort Hamilton","Gowanus","Gravesend","Greenpoint","Kensington","Manhattan Beach","Midwood","Mill Basin","Navy Yard","Park Slope","Prospect Heights","Prospect-Lefferts Gardens","Red Hook","Sea Gate","Sheepshead Bay","South Slope","Sunset Park","Vinegar Hill","Williamsburg","Windsor Terrace"],
  "Manhattan": ["Battery Park City","Chelsea","Chinatown","Civic Center","East Harlem","East Village","Financial District","Flatiron District","Gramercy","Greenwich Village","Harlem","Hell's Kitchen","Inwood","Kips Bay","Little Italy","Lower East Side","Marble Hill","Midtown","Morningside Heights","Murray Hill","NoHo","Nolita","Roosevelt Island","SoHo","Stuyvesant Town","Theater District","Tribeca","Two Bridges","Upper East Side","Upper West Side","Washington Heights","West Village"],
  "Queens": ["Arverne","Astoria","Bay Terrace","Bayside","Bayswater","Belle Harbor","Bellerose","Breezy Point","Briarwood","Cambria Heights","College Point","Corona","Ditmars Steinway","Douglaston","East Elmhurst","Edgemere","Elmhurst","Far Rockaway","Flushing","Forest Hills","Fresh Meadows","Glendale","Hollis","Holliswood","Howard Beach","Jackson Heights","Jamaica","Jamaica Estates","Jamaica Hills","Kew Gardens","Kew Gardens Hills","Laurelton","Little Neck","Long Island City","Maspeth","Middle Village","Neponsit","Ozone Park","Queens Village","Rego Park","Richmond Hill","Ridgewood","Rockaway Beach","Rosedale","South Ozone Park","Springfield Gardens","St. Albans","Sunnyside","Whitestone","Woodhaven","Woodside"],
  "Staten Island": ["Arden Heights","Arrochar","Bay Terrace, Staten Island","Bull's Head","Castleton Corners","Clifton","Concord","Dongan Hills","Eltingville","Emerson Hill","Graniteville","Grant City","Great Kills","Grymes Hill","Howland Hook","Huguenot","Mariners Harbor","Midland Beach","New Brighton","New Dorp","New Dorp Beach","New Springville","Oakwood","Port Richmond","Prince's Bay","Randall Manor","Rosebank","Rossville","Shore Acres","Silver Lake","South Beach","St. George","Stapleton","Todt Hill","Tompkinsville","Tottenville","West Brighton","Westerleigh","Willowbrook"],
};

const CENTERS = {
  "Bronx":         { lat: 40.8448, lng: -73.8648 },
  "Brooklyn":      { lat: 40.6782, lng: -73.9442 },
  "Manhattan":     { lat: 40.7831, lng: -73.9712 },
  "Queens":        { lat: 40.7282, lng: -73.7949 },
  "Staten Island": { lat: 40.5795, lng: -74.1502 },
};

const PRESETS = {
  midtown:      { borough: "Manhattan", neighbourhood: "Midtown",        latitude: 40.7549, longitude: -73.9840, price: 285, minimum_nights: 3, number_of_reviews: 42, reviews_per_month: 1.1, calculated_host_listings_count: 2, availability_365: 210 },
  williamsburg: { borough: "Brooklyn",  neighbourhood: "Williamsburg",   latitude: 40.7081, longitude: -73.9571, price: 175, minimum_nights: 5, number_of_reviews: 64, reviews_per_month: 1.4, calculated_host_listings_count: 1, availability_365: 120 },
  bushwick:     { borough: "Brooklyn",  neighbourhood: "Bushwick",       latitude: 40.6944, longitude: -73.9213, price: 62,  minimum_nights: 2, number_of_reviews: 31, reviews_per_month: 0.8, calculated_host_listings_count: 1, availability_365: 150 },
  hellskitchen: { borough: "Manhattan", neighbourhood: "Hell's Kitchen", latitude: 40.7638, longitude: -73.9918, price: 30,  minimum_nights: 1, number_of_reviews: 40, reviews_per_month: 1.5, calculated_host_listings_count: 5, availability_365: 365 },
  fordham:      { borough: "Bronx",     neighbourhood: "Fordham",        latitude: 40.8448, longitude: -73.8648, price: 95,  minimum_nights: 2, number_of_reviews: 15, reviews_per_month: 0.7, calculated_host_listings_count: 1, availability_365: 180 },
};

/* Mirrors the Pydantic model in the API */
const RULES = {
  latitude:                       { kind: "float", min: -90,  max: 90 },
  longitude:                      { kind: "float", min: -180, max: 180 },
  price:                          { kind: "float", gt: 0 },
  minimum_nights:                 { kind: "int",   min: 1, max: 365 },
  number_of_reviews:              { kind: "int",   min: 0 },
  reviews_per_month:              { kind: "float", min: 0 },
  calculated_host_listings_count: { kind: "int",   min: 0 },
  availability_365:               { kind: "int",   min: 0, max: 365 },
};
const NUMERIC_FIELDS = Object.keys(RULES);

/* ---------- Tiny helpers ---------- */

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));
const clamp = (n, lo, hi) => Math.min(hi, Math.max(lo, n));

const store = {
  get(key) { try { return localStorage.getItem(key); } catch { return null; } },
  set(key, val) { try { localStorage.setItem(key, val); } catch { /* storage unavailable */ } },
  remove(key) { try { localStorage.removeItem(key); } catch { /* ignore */ } },
};

const form = $("#predict-form");
const result = $("#result");

function fire(el, type = "input") { el.dispatchEvent(new Event(type, { bubbles: true })); }

function fmtPct(v) {
  if (v <= 0) return "0%";
  if (v < 0.5) return "<1%";
  return Math.round(v) + "%";
}

/* ==========================================================
   API address + connection status
   ========================================================== */

let apiBase = (store.get(STORE_API) || DEFAULT_API).replace(/\/+$/, "");
const pill = $("#api-pill");
const pillText = $("#api-pill-text");

function setApiStatus(status) {
  pill.dataset.status = status;
  pillText.textContent =
    status === "ok" ? "API connected" :
    status === "off" ? "API unreachable" : "Checking API…";
}

async function checkApi() {
  setApiStatus("checking");
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 4000);
  try {
    const res = await fetch(apiBase + "/", { signal: ctrl.signal });
    setApiStatus(res.ok ? "ok" : "off");
  } catch {
    setApiStatus("off");
  } finally {
    clearTimeout(timer);
  }
}

const dialog = $("#api-dialog");
const apiUrlInput = $("#api-url");
const apiUrlError = $("#api-url-error");

pill.addEventListener("click", () => {
  apiUrlInput.value = apiBase;
  apiUrlError.hidden = true;
  dialog.showModal();
  apiUrlInput.select();
});
$("#api-cancel").addEventListener("click", () => dialog.close());
$("#api-default").addEventListener("click", () => { apiUrlInput.value = DEFAULT_API; });
dialog.addEventListener("click", (e) => { if (e.target === dialog) dialog.close(); });

$("#api-form").addEventListener("submit", (e) => {
  e.preventDefault();
  let url;
  try {
    url = new URL(apiUrlInput.value.trim());
    if (!/^https?:$/.test(url.protocol)) throw new Error("protocol");
  } catch {
    apiUrlError.textContent = "Enter a full address that starts with http:// or https://";
    apiUrlError.hidden = false;
    return;
  }
  apiBase = url.origin + url.pathname.replace(/\/+$/, "");
  store.set(STORE_API, apiBase);
  dialog.close();
  checkApi();
});

window.addEventListener("focus", () => { if (pill.dataset.status !== "ok") checkApi(); });

/* ==========================================================
   Borough chips + neighbourhood combobox
   ========================================================== */

const hoodInput = $("#neighbourhood");
const hoodList = $("#hood-list");
let shownHoods = [];
let activeIdx = -1;

const getBorough = () => $('input[name="neighbourhood_group"]:checked').value;
const setBorough = (name) => {
  const radio = $(`input[name="neighbourhood_group"][value="${name}"]`);
  if (radio) radio.checked = true;
};

function renderHoodList() {
  const q = hoodInput.value.trim().toLowerCase();
  const all = HOODS[getBorough()];
  shownHoods = q
    ? [...all.filter(n => n.toLowerCase().startsWith(q)), ...all.filter(n => !n.toLowerCase().startsWith(q) && n.toLowerCase().includes(q))]
    : all.slice();

  hoodList.replaceChildren();
  if (!shownHoods.length) {
    const li = document.createElement("li");
    li.className = "none";
    li.textContent = `No match in ${getBorough()}`;
    hoodList.append(li);
    activeIdx = -1;
    return;
  }
  shownHoods.forEach((name, i) => {
    const li = document.createElement("li");
    li.id = "hood-opt-" + i;
    li.setAttribute("role", "option");
    const at = q ? name.toLowerCase().indexOf(q) : -1;
    if (at >= 0) {
      const mark = document.createElement("mark");
      mark.textContent = name.slice(at, at + q.length);
      li.append(name.slice(0, at), mark, name.slice(at + q.length));
    } else {
      li.textContent = name;
    }
    li.addEventListener("mousedown", (e) => e.preventDefault());
    li.addEventListener("click", () => chooseHood(name));
    hoodList.append(li);
  });
  setActive(q ? 0 : -1);
}

function setActive(i) {
  activeIdx = i;
  $$("#hood-list li").forEach((li, idx) => li.setAttribute("aria-selected", String(idx === i)));
  if (i >= 0) {
    const el = $("#hood-opt-" + i);
    hoodInput.setAttribute("aria-activedescendant", "hood-opt-" + i);
    el && el.scrollIntoView({ block: "nearest" });
  } else {
    hoodInput.removeAttribute("aria-activedescendant");
  }
}

function openHoods() {
  renderHoodList();
  hoodList.hidden = false;
  hoodInput.setAttribute("aria-expanded", "true");
}
function closeHoods() {
  hoodList.hidden = true;
  hoodInput.setAttribute("aria-expanded", "false");
  hoodInput.removeAttribute("aria-activedescendant");
}
function chooseHood(name) {
  hoodInput.value = name;
  closeHoods();
  showError("neighbourhood", "");
}

hoodInput.addEventListener("focus", openHoods);
hoodInput.addEventListener("input", () => { openHoods(); showError("neighbourhood", ""); });
hoodInput.addEventListener("blur", () => {
  closeHoods();
  const typed = hoodInput.value.trim().toLowerCase();
  const exact = HOODS[getBorough()].find(n => n.toLowerCase() === typed);
  if (exact) hoodInput.value = exact;
  if (hoodInput.value.trim()) validateHood();
});
hoodInput.addEventListener("keydown", (e) => {
  const isOpen = !hoodList.hidden;
  if (e.key === "ArrowDown") {
    e.preventDefault();
    if (!isOpen) openHoods();
    if (shownHoods.length) setActive((activeIdx + 1) % shownHoods.length);
  } else if (e.key === "ArrowUp") {
    e.preventDefault();
    if (!isOpen) openHoods();
    if (shownHoods.length) setActive((activeIdx - 1 + shownHoods.length) % shownHoods.length);
  } else if (e.key === "Enter" && isOpen && activeIdx >= 0) {
    e.preventDefault();
    chooseHood(shownHoods[activeIdx]);
  } else if (e.key === "Escape" && isOpen) {
    e.preventDefault();
    closeHoods();
  }
});

$("#borough-chips").addEventListener("change", () => {
  const b = getBorough();
  if (!HOODS[b].includes(hoodInput.value)) hoodInput.value = "";
  showError("neighbourhood", "");
  if (!hoodList.hidden) renderHoodList();
  moveTo(CENTERS[b].lat, CENTERS[b].lng);
});

/* ==========================================================
   Locator (click or drag to set latitude and longitude)
   ========================================================== */

const LOC = { w: 265, h: 220, latTop: 40.93, latBottom: 40.49, lngLeft: -74.30, lngRight: -73.60 };
const locator = $("#locator");
const pin = $("#pin");
const latInput = $("#latitude");
const lngInput = $("#longitude");
const NS = "http://www.w3.org/2000/svg";

const toX = (lng) => ((lng - LOC.lngLeft) / (LOC.lngRight - LOC.lngLeft)) * LOC.w;
const toY = (lat) => ((LOC.latTop - lat) / (LOC.latTop - LOC.latBottom)) * LOC.h;
const toLng = (x) => LOC.lngLeft + (x / LOC.w) * (LOC.lngRight - LOC.lngLeft);
const toLat = (y) => LOC.latTop - (y / LOC.h) * (LOC.latTop - LOC.latBottom);

function svgEl(name, attrs) {
  const el = document.createElementNS(NS, name);
  for (const k in attrs) el.setAttribute(k, attrs[k]);
  return el;
}

function buildLocator() {
  const grid = $("#locator-grid");
  for (let lat = 40.5; lat <= 40.9; lat += 0.1) {
    const y = toY(lat);
    grid.append(svgEl("line", { x1: 0, x2: LOC.w, y1: y, y2: y }));
  }
  for (let lng = -74.2; lng <= -73.65; lng += 0.1) {
    const x = toX(lng);
    grid.append(svgEl("line", { x1: x, x2: x, y1: 0, y2: LOC.h }));
  }
  const centers = $("#locator-centers");
  for (const [name, c] of Object.entries(CENTERS)) {
    const g = svgEl("g", { "data-borough": name });
    g.append(svgEl("circle", { class: "center-dot", cx: toX(c.lng), cy: toY(c.lat), r: 2.5 }));
    const t = svgEl("text", { class: "center-label", x: toX(c.lng) + 6, y: toY(c.lat) + 3 });
    t.textContent = name;
    g.append(t);
    centers.append(g);
  }
}

function highlightBorough() {
  $$("#locator-centers text").forEach(t => {
    t.classList.toggle("active", t.parentNode.dataset.borough === getBorough());
  });
}

function placePin() {
  const lat = parseFloat(latInput.value);
  const lng = parseFloat(lngInput.value);
  if (Number.isNaN(lat) || Number.isNaN(lng)) return;
  const x = clamp(toX(lng), 6, LOC.w - 6);
  const y = clamp(toY(lat), 6, LOC.h - 6);
  pin.setAttribute("transform", `translate(${x.toFixed(1)} ${y.toFixed(1)})`);
}

function moveTo(lat, lng) {
  latInput.value = (+lat).toFixed(5);
  lngInput.value = (+lng).toFixed(5);
  showError("latitude", ""); showError("longitude", "");
  placePin();
}

function pointerToCoords(e) {
  const r = locator.getBoundingClientRect();
  const x = clamp(((e.clientX - r.left) / r.width) * LOC.w, 0, LOC.w);
  const y = clamp(((e.clientY - r.top) / r.height) * LOC.h, 0, LOC.h);
  return { lat: toLat(y), lng: toLng(x) };
}

let dragging = false;
locator.addEventListener("pointerdown", (e) => {
  dragging = true;
  locator.setPointerCapture(e.pointerId);
  const c = pointerToCoords(e);
  moveTo(c.lat, c.lng);
});
locator.addEventListener("pointermove", (e) => {
  if (!dragging) return;
  pin.classList.add("dragging");
  const c = pointerToCoords(e);
  moveTo(c.lat, c.lng);
});
const endDrag = () => { dragging = false; pin.classList.remove("dragging"); };
locator.addEventListener("pointerup", endDrag);
locator.addEventListener("pointercancel", endDrag);

[latInput, lngInput].forEach(el => el.addEventListener("input", placePin));
$("#reset-coords").addEventListener("click", () => {
  const c = CENTERS[getBorough()];
  moveTo(c.lat, c.lng);
});

/* ==========================================================
   Steppers, sliders, year strip
   ========================================================== */

function decimalsOf(step) {
  const s = String(step);
  return s.includes(".") ? s.split(".")[1].length : 0;
}

$$(".stepper").forEach(box => {
  const input = $("input", box);
  const min = parseFloat(box.dataset.min);
  const max = parseFloat(box.dataset.max);
  const step = parseFloat(box.dataset.step);
  const dec = decimalsOf(step);
  let holdTimer = null, repeatTimer = null;

  function bump(dir) {
    const cur = parseFloat(input.value);
    const base = Number.isNaN(cur) ? (min > 0 ? min : 0) : cur;
    const next = clamp(+(base + dir * step).toFixed(dec), min, max);
    input.value = next;
    fire(input);
  }
  function stopHold() { clearTimeout(holdTimer); clearInterval(repeatTimer); }

  $$("button", box).forEach(btn => {
    const dir = parseInt(btn.dataset.dir, 10);
    btn.addEventListener("pointerdown", (e) => {
      if (e.button !== 0) return;
      bump(dir);
      holdTimer = setTimeout(() => { repeatTimer = setInterval(() => bump(dir), 70); }, 420);
    });
    ["pointerup", "pointerleave", "pointercancel"].forEach(t => btn.addEventListener(t, stopHold));
    btn.addEventListener("click", (e) => { if (e.detail === 0) bump(dir); }); // keyboard activation
  });
});

function paintRange(range) {
  const min = +range.min, max = +range.max;
  const pct = ((clamp(+range.value, min, max) - min) / (max - min)) * 100;
  range.style.setProperty("--pct", pct + "%");
}

function bindPair(numId, rangeId, after) {
  const num = $("#" + numId), range = $("#" + rangeId);
  num.addEventListener("input", () => {
    const v = parseFloat(num.value);
    if (!Number.isNaN(v)) range.value = clamp(v, +range.min, +range.max);
    paintRange(range);
    after && after();
  });
  range.addEventListener("input", () => {
    num.value = range.value;
    fire(num, "change");
    showError(numId, "");
    paintRange(range);
    after && after();
  });
  paintRange(range);
}

// Year strip: 12 month cells fill in as availability grows
const year = $("#year");
for (let i = 0; i < 12; i++) year.append(document.createElement("i"));

function updateYear() {
  const days = clamp(parseInt($("#availability_365").value, 10) || 0, 0, 365);
  const months = (days / 365) * 12;
  $$("i", year).forEach((cell, i) => {
    cell.style.setProperty("--fill", clamp(months - i, 0, 1) * 100 + "%");
  });
  $("#year-note").textContent =
    days === 0 ? "Not open for booking at any point in the year" :
    days >= 365 ? "Open all year" :
    days < 60 ? `${days} days a year` :
    `About ${Math.round(months)} months of the year`;
}

bindPair("price", "price-range");
bindPair("availability_365", "availability-range", updateYear);

/* ==========================================================
   Validation (same rules as the Pydantic model)
   ========================================================== */

function showError(name, msg) {
  const el = $("#err-" + name);
  const input = $("#" + name);
  if (!el) return;
  el.textContent = msg;
  el.hidden = !msg;
  if (input) {
    if (msg) input.setAttribute("aria-invalid", "true");
    else input.removeAttribute("aria-invalid");
  }
}

function checkNumber(name) {
  const rule = RULES[name];
  const raw = $("#" + name).value.trim();
  if (raw === "") return "Enter a number.";
  const n = Number(raw);
  if (!Number.isFinite(n)) return "Enter a number.";
  if (rule.kind === "int" && !Number.isInteger(n)) return "Use a whole number.";
  if (rule.gt !== undefined && !(n > rule.gt)) return `Must be greater than ${rule.gt}.`;
  if (rule.min !== undefined && rule.max !== undefined && (n < rule.min || n > rule.max)) return `Must be between ${rule.min} and ${rule.max}.`;
  if (rule.min !== undefined && n < rule.min) return rule.min === 0 ? "Can't be negative." : `Must be at least ${rule.min}.`;
  if (rule.max !== undefined && n > rule.max) return `Must be at most ${rule.max}.`;
  return "";
}

function validateHood() {
  const value = hoodInput.value.trim();
  let msg = "";
  if (!value) msg = "Choose a neighbourhood.";
  else if (!HOODS[getBorough()].includes(value)) msg = `Pick a ${getBorough()} neighbourhood from the list.`;
  showError("neighbourhood", msg);
  return !msg;
}

NUMERIC_FIELDS.forEach(name => {
  const el = $("#" + name);
  el.addEventListener("blur", () => showError(name, checkNumber(name)));
  el.addEventListener("input", () => { if (el.hasAttribute("aria-invalid") && !checkNumber(name)) showError(name, ""); });
});

function validateAll() {
  let firstBad = null;
  if (!validateHood()) firstBad = hoodInput;
  NUMERIC_FIELDS.forEach(name => {
    const msg = checkNumber(name);
    showError(name, msg);
    if (msg && !firstBad) firstBad = $("#" + name);
  });
  if (firstBad) firstBad.focus();
  return !firstBad;
}

function collectPayload() {
  const p = {
    latitude: Number($("#latitude").value),
    longitude: Number($("#longitude").value),
    price: Number($("#price").value),
    minimum_nights: parseInt($("#minimum_nights").value, 10),
    number_of_reviews: parseInt($("#number_of_reviews").value, 10),
    reviews_per_month: Number($("#reviews_per_month").value),
    calculated_host_listings_count: parseInt($("#calculated_host_listings_count").value, 10),
    availability_365: parseInt($("#availability_365").value, 10),
    neighbourhood_group: getBorough(),
    neighbourhood: hoodInput.value.trim(),
  };
  return p;
}

/* ==========================================================
   Result rendering
   ========================================================== */

const bullet = $("#bullet");
const linesEl = $("#lines");
const errorBox = $("#error-box");

function metaFor(key) {
  return CLASS_META[key] || { label: key, short: (key || "?").charAt(0).toUpperCase(), color: "var(--muted)" };
}

function setState(state) { result.dataset.state = state; }

function renderIdleLines() {
  linesEl.replaceChildren();
  CLASS_ORDER.forEach(key => linesEl.append(buildLine(key, null, false)));
}

function buildLine(key, pct, top) {
  const m = metaFor(key);
  const li = document.createElement("li");
  li.className = "line" + (top ? " top" : "");
  li.style.setProperty("--lc", m.color);
  li.innerHTML =
    '<span class="mini"></span><span class="name"></span><span class="pct"></span>' +
    '<span class="track"><span class="fill"></span></span>';
  $(".mini", li).textContent = m.short;
  $(".name", li).textContent = m.label;
  $(".pct", li).textContent = pct === null ? "–" : "0%";
  return li;
}

function countUp(el, to) {
  if (reduceMotion) { el.textContent = fmtPct(to); return; }
  const t0 = performance.now(), dur = 950;
  (function frame(t) {
    const k = Math.min(1, (t - t0) / dur);
    const eased = 1 - Math.pow(1 - k, 3);
    el.textContent = k < 1 ? Math.round(to * eased) + "%" : fmtPct(to);
    if (k < 1) requestAnimationFrame(frame);
  })(t0);
}

function hideError() { errorBox.hidden = true; }

function showApiError(title, body) {
  setState("error");
  bullet.textContent = "!";
  $("#verdict-label").textContent = "No prediction yet";
  $("#verdict-sub").textContent = "Fix the issue below and try again.";
  $("#error-title").textContent = title;
  $("#error-body").textContent = body;
  errorBox.hidden = false;
  // restart the shake
  errorBox.style.animation = "none"; void errorBox.offsetWidth; errorBox.style.animation = "";
  renderIdleLines();
}

/* The API can answer in either of two shapes; both are accepted:
   new:  { predicted_room_type, probabilities: { "Private room": 0.45, ... } }
   old:  { Predicted_room_type, Probability: [..], Classes?: [..] }          */
function normaliseResponse(data) {
  if (!data || typeof data !== "object") return null;
  const label = data.predicted_room_type ?? data.Predicted_room_type;
  if (typeof label !== "string") return null;

  const isMap = (o) => o && typeof o === "object" && !Array.isArray(o);
  const map = isMap(data.probabilities) ? data.probabilities
            : isMap(data.Probabilities) ? data.Probabilities : null;
  if (map) {
    const keys = Object.keys(map);
    if (!keys.length) return null;
    return { label, classes: keys, probs: keys.map(k => Number(map[k])) };
  }
  if (Array.isArray(data.Probability)) {
    const classes = Array.isArray(data.Classes) && data.Classes.length === data.Probability.length
      ? data.Classes : CLASS_ORDER;
    return { label, classes, probs: data.Probability.map(Number) };
  }
  return null;
}

function renderResult(norm, payload, ms) {
  const entries = norm.probs
    .map((p, i) => ({ key: norm.classes[i] || `Class ${i + 1}`, pct: p * 100 }))
    .sort((a, b) => b.pct - a.pct);

  const winnerKey = norm.label;
  const winner = metaFor(winnerKey);
  const top = entries[0], second = entries[1];
  const gap = second ? top.pct - second.pct : 100;

  setState("done");
  result.style.setProperty("--c", winner.color);
  hideError();

  bullet.textContent = winner.short;
  bullet.classList.remove("pop"); void bullet.offsetWidth; bullet.classList.add("pop");

  $("#verdict-label").textContent = winner.label;
  $("#verdict-sub").textContent =
    top.pct < 50 || gap < 10
      ? `Close call: ${metaFor(top.key).label} edges out ${metaFor(second.key).label} by ${Math.max(1, Math.round(gap))} ${Math.round(gap) <= 1 ? "point" : "points"}.`
      : top.pct >= 70
        ? `${Math.round(top.pct)}% confidence. Strong match.`
        : `${Math.round(top.pct)}% confidence. Likely, not certain.`;

  const latency = $("#latency");
  latency.textContent = `${Math.round(ms)} ms`;
  latency.hidden = false;

  linesEl.replaceChildren();
  entries.forEach((e, i) => {
    const li = buildLine(e.key, e.pct, i === 0);
    linesEl.append(li);
    countUp($(".pct", li), e.pct);
    requestAnimationFrame(() => requestAnimationFrame(() => {
      $(".fill", li).style.width = clamp(e.pct, 0, 100) + "%";
    }));
  });

  $("#payload").textContent = JSON.stringify(payload, null, 2);
  $("#payload-box").hidden = false;

  addHistory({ payload, key: winnerKey, pct: top.pct });

  if (window.matchMedia("(max-width: 960px)").matches) {
    result.scrollIntoView({ behavior: reduceMotion ? "auto" : "smooth", block: "start" });
  }
}

/* ==========================================================
   Predict
   ========================================================== */

const submitBtn = $("#submit-btn");

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  if (!validateAll()) return;

  const payload = collectPayload();
  submitBtn.disabled = true;
  hideError();
  setState("loading");
  bullet.textContent = "…";
  $("#verdict-label").textContent = "Reading the listing…";
  $("#verdict-sub").textContent = "Asking the model.";
  linesEl.replaceChildren();
  $("#latency").hidden = true;

  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 20000);
  const t0 = performance.now();

  try {
    const res = await fetch(apiBase + "/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
      signal: ctrl.signal,
    });
    setApiStatus("ok");

    if (!res.ok) {
      let detail = null;
      try { detail = await res.json(); } catch { /* not JSON */ }

      if (res.status === 422 && detail && Array.isArray(detail.detail)) {
        const lines = detail.detail.map(d => {
          const field = Array.isArray(d.loc) ? d.loc[d.loc.length - 1] : "input";
          if (NUMERIC_FIELDS.includes(field) || field === "neighbourhood") showError(field, d.msg);
          return `${field}: ${d.msg}`;
        });
        showApiError("The API rejected some values", lines.join("\n"));
      } else {
        showApiError(
          `The server returned an error (${res.status})`,
          "The model could not score this listing. Check the terminal running uvicorn for the full traceback."
        );
      }
      return;
    }

    const data = await res.json();
    const norm = normaliseResponse(data);
    if (!norm || norm.probs.some(p => !Number.isFinite(p))) {
      showApiError(
        "The response was not what this page expects",
        'Expected JSON with "predicted_room_type" and a "probabilities" object (or "Predicted_room_type" and a "Probability" list). Check what your /predict route returns.'
      );
      return;
    }
    renderResult(norm, payload, performance.now() - t0);

  } catch (err) {
    if (err.name === "AbortError") {
      showApiError("The API took too long to answer", "No response after 20 seconds. The model may still be loading. Try again in a moment.");
    } else {
      setApiStatus("off");
      showApiError(
        "Can't reach the API",
        `Nothing answered at ${apiBase}. Start the server with "uvicorn main:app --reload", or change the address with the status button at the top.`
      );
    }
  } finally {
    clearTimeout(timer);
    submitBtn.disabled = false;
  }
});

/* ==========================================================
   History
   ========================================================== */

let recent = [];
try { recent = JSON.parse(store.get(STORE_HISTORY) || "[]"); } catch { recent = []; }
if (!Array.isArray(recent)) recent = [];

function addHistory(entry) {
  recent.unshift(entry);
  recent = recent.slice(0, HISTORY_MAX);
  store.set(STORE_HISTORY, JSON.stringify(recent));
  renderHistory();
}

function renderHistory() {
  const list = $("#history-list");
  list.replaceChildren();
  $("#history-empty").hidden = recent.length > 0;
  $("#clear-history").hidden = recent.length === 0;

  recent.forEach(h => {
    const m = metaFor(h.key);
    const p = h.payload;
    const li = document.createElement("li");
    const btn = document.createElement("button");
    btn.type = "button";
    btn.style.setProperty("--lc", m.color);
    btn.innerHTML = '<span class="h-dot"></span><span class="h-main"><span class="h-title"></span><span class="h-sub"></span></span><span class="h-pct"></span>';
    $(".h-title", btn).textContent = m.label;
    $(".h-sub", btn).textContent = `${p.neighbourhood}, ${p.neighbourhood_group} · $${p.price} · ${p.minimum_nights}-night min`;
    $(".h-pct", btn).textContent = fmtPct(h.pct);
    btn.addEventListener("click", () => fillForm({
      borough: p.neighbourhood_group, neighbourhood: p.neighbourhood,
      latitude: p.latitude, longitude: p.longitude, price: p.price,
      minimum_nights: p.minimum_nights, number_of_reviews: p.number_of_reviews,
      reviews_per_month: p.reviews_per_month,
      calculated_host_listings_count: p.calculated_host_listings_count,
      availability_365: p.availability_365,
    }));
    li.append(btn);
    list.append(li);
  });
}

$("#clear-history").addEventListener("click", () => {
  recent = [];
  store.remove(STORE_HISTORY);
  renderHistory();
});

/* ==========================================================
   Presets, fill, reset, copy
   ========================================================== */

function syncAll() {
  paintRange($("#price-range"));
  paintRange($("#availability-range"));
  const price = parseFloat($("#price").value);
  if (!Number.isNaN(price)) $("#price-range").value = clamp(price, 10, 1000);
  const av = parseFloat($("#availability_365").value);
  if (!Number.isNaN(av)) $("#availability-range").value = clamp(av, 0, 365);
  paintRange($("#price-range"));
  paintRange($("#availability-range"));
  updateYear();
  placePin();
  highlightBorough();
}

function fillForm(v) {
  setBorough(v.borough);
  hoodInput.value = v.neighbourhood;
  NUMERIC_FIELDS.forEach(name => { $("#" + name).value = v[name]; showError(name, ""); });
  showError("neighbourhood", "");
  syncAll();
  $$(".stop").forEach(s => {
    s.classList.remove("flash"); void s.offsetWidth; s.classList.add("flash");
  });
}

$$(".preset").forEach(btn => btn.addEventListener("click", () => {
  fillForm(PRESETS[btn.dataset.preset]);
  submitBtn.focus({ preventScroll: true });
}));

$("#borough-chips").addEventListener("change", highlightBorough);

$("#reset-form").addEventListener("click", () => {
  form.reset();
  hoodInput.value = "";
  NUMERIC_FIELDS.forEach(name => showError(name, ""));
  showError("neighbourhood", "");
  setTimeout(syncAll, 0);
});

$("#copy-payload").addEventListener("click", async (e) => {
  const btn = e.currentTarget;
  const text = $("#payload").textContent;
  try {
    await navigator.clipboard.writeText(text);
    btn.textContent = "Copied";
  } catch {
    const range = document.createRange();
    range.selectNodeContents($("#payload"));
    const sel = window.getSelection();
    sel.removeAllRanges(); sel.addRange(range);
    btn.textContent = "Selected. Press Ctrl+C";
  }
  setTimeout(() => { btn.textContent = "Copy JSON"; }, 1800);
});

/* ==========================================================
   Boot
   ========================================================== */

buildLocator();
renderIdleLines();
renderHistory();
syncAll();
checkApi();

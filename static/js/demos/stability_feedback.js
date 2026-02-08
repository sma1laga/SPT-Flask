(() => {
  const demo = document.querySelector("[data-stability-defaults]");
  if (!demo) return;

  const defaults = JSON.parse(demo.dataset.stabilityDefaults || "{}");
  const computeUrl = demo.dataset.computeUrl || "./compute";

  const $ = (id) => document.getElementById(id);

  let inFlight = null;
  let debounceTimer = null;
  let SEQ = 0;
  const sanitizeNumber = (value, fallback) => {
    const parsed = Number.parseFloat(value);
    return Number.isFinite(parsed) ? parsed : fallback;
  };

  function send() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(async () => {
      if (inFlight) inFlight.abort();
      inFlight = new AbortController();

      const mySeq = ++SEQ;
      const body = {
        a: sanitizeNumber($("a").value, defaults.a),
        b: sanitizeNumber($("b").value, defaults.b),
        K: sanitizeNumber($("K").value, defaults.K),
        seq: mySeq
      };

      try {
        const resp = await fetch(computeUrl, {
          method: "POST",
          headers: {"Content-Type":"application/json"},
          body: JSON.stringify(body),
          signal: inFlight.signal
        });
        if (!resp.ok) return;
        const data = await resp.json();

        if (data.seq !== undefined && data.seq < SEQ) return;

        if (data.image) $("plot").src = data.image;
        if (Number.isFinite(data.a)) $("a").value = data.a;
        if (Number.isFinite(data.b)) $("b").value = data.b;
        if (Number.isFinite(data.K)) $("K").value = data.K;
      } catch (e) {
        // ignore aborts
      } finally {
        inFlight = null;
      }
    }, 80);
  }

  ["a","b","K"].forEach(id => {
    $(id).addEventListener("input", send);
    $(id).addEventListener("change", send);
    $(id).addEventListener("keyup", e => { if (e.key === "Enter") send(); });
  });

  window.addEventListener("load", send);

  $("run").addEventListener("click", send);

  $("reset").addEventListener("click", () => {
    ["a","b","K"].forEach(id => { $(id).value = defaults[id]; });
    send();
  });
})();
(() => {
  const demo = document.querySelector("[data-sampling-defaults]");
  if (!demo) return;

  const defaults = JSON.parse(demo.dataset.samplingDefaults || "{}");
  const computeUrl = demo.dataset.computeUrl || "./compute";

  const $ = (id) => document.getElementById(id);

  let inFlight = null;
  let debounceTimer = null;
  let SEQ = 0;

  function payload() {
    return {
      w_g_over_pi: parseFloat($("wg_over_pi").value),
      w_a_over_pi: parseFloat($("wa_over_pi").value),
      show_xt: $("show_xt").checked,
      show_xa: $("show_xa").checked,
      show_yt: $("show_yt").checked,
      show_partials: $("show_partials").checked,
      show_X: $("show_X").checked,
      show_Xa: $("show_Xa").checked,
      show_Y: $("show_Y").checked,
      seq: ++SEQ
    };
  }

  function recompute() {
    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(async () => {
      if (inFlight) inFlight.abort();
      inFlight = new AbortController();

      const body = payload();

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
      } catch (e) {
        // ignore abort
      } finally {
        inFlight = null;
      }
    }, 80);
  }

  ["wg_over_pi","wa_over_pi","show_xt","show_xa","show_yt","show_partials","show_X","show_Xa","show_Y"].forEach(id => {
    $(id).addEventListener("input",  recompute);
    $(id).addEventListener("change", recompute);
  });

  $("run").addEventListener("click", recompute);

  $("reset").addEventListener("click", () => {
    $("wg_over_pi").value = defaults.w_g_over_pi;
    $("wa_over_pi").value = defaults.w_a_over_pi;
    $("show_xt").checked = Boolean(defaults.show_xt);
    $("show_xa").checked = Boolean(defaults.show_xa);
    $("show_yt").checked = Boolean(defaults.show_yt);
    $("show_partials").checked = Boolean(defaults.show_partials);
    $("show_X").checked = Boolean(defaults.show_X);
    $("show_Xa").checked = Boolean(defaults.show_Xa);
    $("show_Y").checked = Boolean(defaults.show_Y);
    recompute();
  });

  window.addEventListener("load", recompute);
})();
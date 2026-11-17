# 09: Convert legacy recipes and replay archived runs

**What to build:** An upstream asset owner can package/import a supplied recipe, convert supported
GUI projects with explicit compatibility diagnostics, and reproduce an archived run using frozen
configuration and unchanged files after the original operation expires.

**Blocked by:** 08: Inspect and export measurement results.

**Status:** needs-triage

**Spec user stories:** 80, 81.

- [ ] Deliver portable recipe inspection through validation and versioned recipe import/export with
  explicit asset-reference resolution; existing template/calibration portable contracts remain
  upstream-owned.

- [ ] Deliver one-way conversion of supported GUI project data to the new recipe/assets with stable
  dependency ports and diagnostics for unsupported fields or incompatible legacy content.

- [ ] New public formats use versioned JSON and declared numeric/image payloads and do not require
  arbitrary pickle objects, GUI initialization or old V2 session/publication semantics.

- [ ] Deliver replay from an upstream-retained frozen manifest containing configuration, verified
  input references/digests and algorithm/runtime versions, without relying on a live historical job
  record.

- [ ] Replay invokes the same recipe and atomic execution chain and returns new operation identity
  and isolated outputs without modifying original assets.

- [ ] Return explicit changed/missing-input and incompatible-version outcomes; replay does not
  reconstruct business mutation history or infer replacement asset versions.

- [ ] Verify portable round trips, supported legacy conversion with unsupported-field diagnostics,
  and archived-run replay after operational expiration using public requests.

- [ ] Demonstrate original and replayed numerical equivalence for the same supported
  algorithm/configuration/input identity and preservation of upstream files.


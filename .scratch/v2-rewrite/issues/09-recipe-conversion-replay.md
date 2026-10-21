# 09: Convert legacy recipes and replay archived runs

**What to build:** An upstream asset owner can package/import a supplied recipe, convert supported
GUI projects with explicit compatibility diagnostics, and reproduce an archived run using frozen
configuration and unchanged files after the original operation expires.

**Blocked by:** 08: Inspect and export measurement results.

**Status:** resolved

**Scheduling note:** Independent version identity, packaging, conversion and replay implementation
can proceed using the completed 07 recipe/assets seam. Ticket 08 remains a blocker for final
interoperability verification, integration and resolution; its result/export implementation is reused.

**Spec user stories:** 80, 81.

- [x] Deliver portable recipe inspection through validation and versioned recipe import/export with
  explicit asset-reference resolution; existing template/calibration portable contracts remain
  upstream-owned.

- [x] Deliver one-way conversion of supported GUI project data to the new recipe/assets with stable
  dependency ports and diagnostics for unsupported fields or incompatible legacy content.

- [x] New public formats use versioned JSON and declared numeric/image payloads and do not require
  arbitrary pickle objects, GUI initialization or old V2 session/publication semantics.

- [x] Deliver replay from an upstream-retained frozen manifest containing configuration, verified
  input references/digests and algorithm/runtime versions, without relying on a live historical job
  record.

- [x] Replay invokes the same recipe and atomic execution chain and returns new operation identity
  and isolated outputs without modifying original assets.

- [x] Return explicit changed/missing-input and incompatible-version outcomes; replay does not
  reconstruct business mutation history or infer replacement asset versions.

- [x] Verify portable round trips, supported legacy conversion with unsupported-field diagnostics,
  and archived-run replay after operational expiration using public requests.

- [x] Demonstrate original and replayed numerical equivalence for the same supported
  algorithm/configuration/input identity and preservation of upstream files.
## Answer

Implemented versioned embedded/reference recipe packages, explicit alias/asset relocation and safe
one-way legacy GUI/core conversion with actionable compatibility diagnostics. Imports preserve
original asset content and explicit upstream provenance/version values without inferred identities.
Archive members, numerical payload storage, finite JSON, images and non-pickle formats are bounded
and verified before producing a complete parent output set. Stable typed dependency ports and
legacy coordinate/probe conventions are translated into the existing recipe executor.

Completed runs record actual source/runtime identity through one shared helper. Frozen export
retains the resolved configuration, expected input digests, producer version and numerical/business
record. Replay validates those identities and invokes the same recipe/atomic chain independently
of historical operational state, with new identities and isolated outputs. Changed/missing inputs,
unknown/incompatible producer identities and altered configuration produce explicit outcomes.
Decoder cache identity includes actual codec/library versions and decode parameters, so upgrades
cannot reuse pixels prepared by a different external binary for the same encoded source.

Verification: all 153 merged public HTTP/boundary tests passed on 2026-10-06 (102.48s), including
expiry-independent replay, original/replayed numerical equivalence, JSON report/frozen export/replay/
visualization interoperability, actual relocated image/template/calibration execution, explicit
provenance preservation, unsupported conversion diagnostics, unsafe package rejection and external
codec cache replacement. The only warning is intentionally emitted while constructing a duplicate
archive member for its rejection case. The implementer reported final 29 portable HTTP cases passing
on Python 3.11 and Python 3.9, plus the codec boundary regression; earlier full Python 3.9 verification
passed 152 cases before the final provenance case. No upstream file or top-level README difference
was staged. Ticket 10's final integration gate is now unblocked.

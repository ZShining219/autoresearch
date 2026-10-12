# Traffic Signal Control — Theory Primer

A self-contained explainer of the signal-control theory behind LibSignal: intersection
movements, conflicts, SUMO's signal encoding, the **NEMA dual-ring / barrier** standard, how
the 8-phase action space used by MPLight / CoLight / PressLight maps onto it, and a
**cross-network audit** of whether every map in the repo follows the convention.

Everything here is grounded in this repo's actual config
(`configs/tsc/mplight.yml → signal_config`) and all 15 network files under
`data/raw_data/` (with `grid4x4` and `manhattan_28x7` as the worked examples).

> This reference doc **accompanies the grid4x4 TLS fix (PR #6)**: it explains the
> signal-control theory behind that change and audits every network in the repo. It stands on
> its own as a primer, independent of the specific code diff.
>
> **Reading path:** §1–§3 build the mechanics (movements → conflicts → SUMO encoding);
> §4–§5 introduce NEMA and show LibSignal's action space *is* the NEMA set; §6 diagnoses the
> grid4x4 bug; §7–§8 generalize (protected/permissive/broken, then the all-maps audit);
> §9 ties back to the realism benchmark.

---

## 1. An intersection has 12 movements

A standard 4-way intersection has four approaches — **N**orth, **E**ast, **S**outh, **W**est —
and from each approach a vehicle can go **R**ight, **T**hrough, or **L**eft. That's
`4 approaches × 3 turns = 12 movements`.

```
                              N leg
                enter from North, heading SOUTH ↓
                          NR    NT    NL
                           ▼     ▼     ▼
          ┌────────────────────────────────────────┐
     WR ► │                                          │ ◄ ER
     WT ► │               CONFLICT BOX               │ ◄ ET
     WL ► │            (paths cross here)            │ ◄ EL
          └────────────────────────────────────────┘
                           ▲     ▲     ▲
                          SL    ST    SR
                enter from South, heading NORTH ↑
                              S leg

   • Map convention: N up, E right — so the left/right sides of the box are the
     West / East legs. Straight arrows show each approach's travel direction.
   • A movement is  <approach><turn>:  e.g. NT = enters from North, goes Through
     (southbound);  NL = enters from North, turns Left (ends up eastbound).
   • The four right turns (NR/ER/SR/WR) are the always-allowed "right-on-red" set.
```

LibSignal indexes these 12 movements in a fixed order
(`configs/tsc/mplight.yml`, comment under `cologne3`):

| idx | 0  | 1  | 2  | 3  | 4  | 5  | 6  | 7  | 8  | 9  | 10 | 11 |
|-----|----|----|----|----|----|----|----|----|----|----|----|----|
| mv  | NR | NT | NL | ER | ET | EL | SR | ST | SL | WR | WT | WL |

**Key fact:** the four **right turns** (idx `0,3,6,9` = NR/ER/SR/WR) are treated as
*always permitted* ("right turn on red"). You'll see this below: they never appear in the
phase definitions, and the grid4x4 junctions are literally typed
`traffic_light_right_on_red` in the `.net.xml`.

---

## 2. Which movements conflict?

Two movements **conflict** if their paths physically cross inside the intersection box.
A signal plan must never give a bright green to two conflicting movements at once.

The safe combinations follow three rules:

1. **Opposing throughs are compatible.** N-through and S-through run head-on-parallel,
   they never cross → safe together. Same for E-through + W-through.
2. **Opposing lefts are compatible** *(protected-left design)*. N-left and S-left both
   curve to their right side of the box; by convention they're run together as a
   "protected left" phase. (Their swept paths *do* touch in the middle — SUMO's conflict
   matrix even flags it — but this is the accepted NEMA protected-left pattern; see §5.)
3. **You may not mix the two streets.** N–S movements and E–W movements are separated by a
   **barrier** (below) — you never serve N-S and E-W green at the same time, because that's
   where the dangerous right-angle "T-bone" crossings happen.

Visually, the difference between a safe pairing and a forbidden one:

```
   COMPATIBLE — N & S throughs           CONFLICTING — N-through × E-through
        ↓     ↑                                     ↓
        ↓     ↑                             ← ← ← ← ╳ ← ← ← ←   (E→W through)
        ↓     ↑                                     ↓
   two parallel streams,                    the streams meet at the ╳ in the
   they never touch  ✔                      middle of the box → T-bone  ✗
```

---

## 3. SUMO's signal encoding (how a phase is written)

In SUMO a traffic light is a `<tlLogic>` with a list of `<phase>` elements. Each phase is a
**state string** — one character per *connection* (movement lane) at that junction.

grid4x4 junctions have **36 connections**: 12 movements (4 approaches × R/T/L), and each
movement is a lane that makes 3 lane→lane links, so `12 × 3 = 36` chars. The characters:

| char | meaning (authoritative SUMO semantics) |
|------|-----------------------------------------|
| `G`  | **green, priority** — go, you have right of way |
| `g`  | green, **minor/permissive** — *yield* to priority foes and filter through gaps; **no mandatory stop**. SUMO's "green minor" (e.g. a permissive left allowed alongside oncoming straight). |
| `r`  | **red** — stop |
| `y`  | yellow |
| `u`  | red+yellow (about to turn green) |
| `s`  | green, **right-on-red** — must come to a **full stop first** (treated as a priority stop, `jmStopSignWait`), then go when a gap is clear. Only generated on `traffic_light_right_on_red` junctions. |
| `o` / `O` | off (blinking-yield / no-signal) |

⚠️ **`g` vs `s` — both yield, but only `s` requires a full stop:**
- `g` (green-minor) = **yield without stopping** — slow, give way to priority foes, filter
  through gaps. Models a *permissive left* (or any give-way movement). ≈ a **yield sign**.
- `s` (right-on-red) = **mandatory full stop, then go** when clear. Models a *right turn on
  red*. ≈ a **stop sign** (the "s" helps you remember). SUMO only emits it on
  `traffic_light_right_on_red` junctions.
- So the correct char for right-turn-on-red is **`s`**, not `g` — `g` would skip the legally
  required stop.

⚠️ **Don't confuse two different uses of the letter "s":**
- In a **connection's `dir` attribute**, `s` = *straight/through* (vs `l`eft, `r`ight, `t`urnaround).
- In a **signal state string**, `s` = *right-on-red green-must-stop*. Putting this on a
  **through or left** lane (instead of a right turn) is exactly **the grid4x4 bug** (see §6).

### Reading a real grid4x4 state string

**Where do the 36 characters come from?** 

- A0 has **4 incoming roads** — one per approach: N (`A1A0`), E (`B0A0`), S (`bottom0A0`),
  W (`left0A0`). Each road has **3 lanes**: `lane 0 = right`, `lane 1 = through`, `lane 2 = left`.

Take one concrete movement: the **North approach's right-turn lane**. Its actual connections
in the network file are:

```
   from A1A0 lane 0  →  A0left0 lane 0     (dir=r, linkIndex 0)
   from A1A0 lane 0  →  A0left0 lane 1     (dir=r, linkIndex 1)
   from A1A0 lane 0  →  A0left0 lane 2     (dir=r, linkIndex 2)
```

Same source lane, turning right onto the same road (`A0left0`) — but that road has **3
lanes**, so a right-turning car could land in any of them. SUMO makes a **separate connection
per destination lane** and gives **each its own character** in the state string. So this one
movement (North's right turn) eats up **3 chars** — always identical (`GGG` or `rrr`, never
`Grr`), because they share the *same physical signal head* (one light can't be green for one
landing lane and red for another).

Now scale up: each approach has 3 lanes (right, through, left), and **each** fans out to 3
landing lanes:

```
   one approach (3 lanes):
      lane 0  right    → 3 landing lanes → 3 chars   ("rrr" or "GGG")
      lane 1  through  → 3 landing lanes → 3 chars   ("sss" or "GGG")
      lane 2  left     → 3 landing lanes → 3 chars   (left-turn signal)
                                            ─────────
                                            9 chars per approach

   4 approaches (N, E, S, W)  ×  9   =   36-char state string
```

So the 36 chars are **4 approach-blocks of 9**, and each block is **3 lanes × 3 identical
chars** = `[RRR][TTT][LLL]`.

Example — grid4x4 **A0 phase 0** (the `NT_ST` phase). Raw string:
`GGGGGGrrrGGGrrrrrrGGGGGGrrrGGGrrrrrr`. Split into the 4 approach-blocks:

```
   approach     9 chars      right  through  left
   ──────────   ─────────    ─────  ───────  ────
   N  (idx 0–8)   GGGGGGrrr     G       G       r
   E  (idx 9–17)  GGGrrrrrr     G       r       r
   S  (idx 18–26) GGGGGGrrr     G       G       r
   W  (idx 27–35) GGGrrrrrr     G       r       r
```

Reading it: **North and South get their through lanes green** (that's the `NT_ST` phase —
the two opposing N–S throughs); East and West throughs/lefts are red; **right turns are green
on every approach** (right-on-red); all left turns red. Clean. ✅

---

## 4. NEMA: the real-world standard this all mirrors

**NEMA** = *National Electrical Manufacturers Association* — the US standard (TS-1 / TS-2)
that essentially every American signal controller implements. "NEMA phasing" means the
**dual-ring, 8-phase, ring-and-barrier** scheme.

### The ring-and-barrier diagram

```
                         BARRIER
                            ║
   Ring 1:   Ø1      Ø2     ║     Ø3      Ø4
   Ring 2:   Ø5      Ø6     ║     Ø7      Ø8
                            ║
            └── one street ─┘   └── cross street ──┘
                (e.g. E–W)          (e.g. N–S)
```

Rules that make it safe:
- **Odd phases (1,3,5,7) = protected LEFT turns. Even phases (2,4,6,8) = THROUGH movements.**
- A controller runs **one phase from Ring 1 + one phase from Ring 2**, and **both must be on
  the same side of the barrier.** That guarantees you never serve conflicting streets.
- The **barrier** is the hard wall between the two streets — crossing it (serving N-S and E-W
  together) is exactly the T-bone case, so it's forbidden.
- **Ø2 and Ø6** are the *coordinated* phases — the mainline opposing throughs used for
  "green-wave" progression along an arterial.

One common FHWA/ITE direction assignment (labels vary by agency; the **structure** is the
invariant that matters):

```
   Ring 1:  Ø1 EBL   Ø2 WBT  ║  Ø3 SBL   Ø4 NBT
   Ring 2:  Ø5 WBL   Ø6 EBT  ║  Ø7 NBL   Ø8 SBT
            └──── E–W ───────┘  └──── N–S ───────┘
```

Legal concurrent pairs (one per ring, same barrier side):
- E–W side: (Ø1,Ø5), (Ø1,Ø6), (Ø2,Ø5), (Ø2,Ø6)
- N–S side: (Ø3,Ø7), (Ø3,Ø8), (Ø4,Ø7), (Ø4,Ø8)

That's **eight** legal pairs — and they are exactly the eight phases §5 shows LibSignal using.
"Pick one phase from each ring, same side of the barrier" *is* the rule that generates the
action space.

---

## 5. LibSignal's 8-phase action space **is** the NEMA legal set

This is the punchline. The RL agents (MPLight, CoLight, PressLight, MaxPressure) don't pick
raw green lights — they pick one of **8 phases**, defined in
`configs/tsc/mplight.yml → signal_config`. For **grid4x4**:

```yaml
grid4x4:
  # phases: ['NT_ST','NL_SL','NT_NL','ST_SL','ET_WT','EL_WL','WT_WL','ET_EL']
  phase_pairs: [[1,7],[2,8],[1,2],[7,8],[4,10],[5,11],[10,11],[4,5]]
```

Decoding each pair with the movement index from §1:

| # | phase_pair | movements | name | NEMA equivalent |
|---|-----------|-----------|------|-----------------|
| 0 | `[1,7]`   | NT + ST | **NT_ST** — opposing N-S throughs | Ø4+Ø8 |
| 1 | `[2,8]`   | NL + SL | **NL_SL** — opposing N-S lefts | Ø3+Ø7 |
| 2 | `[1,2]`   | NT + NL | **NT_NL** — all of North approach | Ø4+Ø7 |
| 3 | `[7,8]`   | ST + SL | **ST_SL** — all of South approach | Ø3+Ø8 |
| 4 | `[4,10]`  | ET + WT | **ET_WT** — opposing E-W throughs | Ø2+Ø6 |
| 5 | `[5,11]`  | EL + WL | **EL_WL** — opposing E-W lefts | Ø1+Ø5 |
| 6 | `[10,11]` | WT + WL | **WT_WL** — all of West approach | Ø2+Ø5 |
| 7 | `[4,5]`   | ET + EL | **ET_EL** — all of East approach | Ø1+Ø6 |

Notice:
- Phases 0–3 are the **N–S barrier group**; phases 4–7 are the **E–W barrier group**. Exactly
  the two sides of the NEMA barrier.
- The 8 phases are precisely the 8 NEMA legal concurrent pairs. **The RL action space is
  NEMA-safe by construction** — an agent physically cannot select a conflicting green.
- Rights (idx 0,3,6,9) never appear → always-permitted (right-on-red).

Each phase is the set of movements that are green **together**. Writing `X→Y` for "enters
from leg X, exits toward leg Y" (so the four left turns are `N→E`, `S→W`, `E→S`, `W→N`):

```
  ┌─ N–S barrier group (phases 0–3) ──┐   ┌─ E–W barrier group (phases 4–7) ──┐
  0  NT_ST   N→S + S→N   opposing throughs   4  ET_WT   E→W + W→E   opposing throughs
  1  NL_SL   N→E + S→W   opposing lefts       5  EL_WL   E→S + W→N   opposing lefts
  2  NT_NL   N→S + N→E   whole N approach      6  WT_WL   W→E + W→N   whole W approach
  3  ST_SL   S→N + S→W   whole S approach      7  ET_EL   E→W + E→S   whole E approach
```

Why the two "opposing" phase types are safe — they look alarming but the paths never cross:

```
   Phase 0  NT_ST  (throughs)          Phase 1  NL_SL  (protected lefts)
            N                                    N
            ▼  N→S                               ╰─►  N→E  (exits East)
     ───────┼───────                     ────────┼────────
       S→N  ▲                            S→W  ◄─╮
            S                                    S
   antiparallel straight streams,       each left curves into its OWN
   never meet  ✔                        quadrant — mirror images, never meet ✔
```

---

## 6. The grid4x4 defect, in these terms

**Master (the old, merged PR #5 network)** wrote the left-turn lanes with the `s`
signal-state in the green phases. Decoding master A0 phase 0:

```
master A0 ph0 = rrrGGGsss | rrrrrrsss | rrrGGGsss | rrrrrrsss
                R=r T=G L=s   ...
```
→ every left lane got `s` = "green, but must stop first" (a *right-on-red* semantic applied
to **left** turns). In the GUI this rendered as the odd **purple** left-turn lanes you saw —
"left purple, center green, right red." Semantically wrong: lefts were being treated as
yield-and-go rather than protected/red.

**PR #6** replaces every intersection's phases with manhattan_28x7's strings, which use only
clean `G`/`r`:

```
PR#6 A0 ph0 = GGGGGGrrr | GGGrrrrrr | GGGGGGrrr | GGGrrrrrr
              R=G T=G L=r   ...
```
→ no `s` anywhere (0 vs master's 96 per intersection), lefts are properly red when not
served. Verified byte-for-byte equal to `manhattan_28x7 intersection_1_1`'s 8 green phases.

So the "fix" is: **remove the bogus permissive-`s` on left lanes; use standard protected
NEMA phases.** The only thing a conflict-checker still flags — opposing lefts sharing a phase
(NL_SL, EL_WL) — is **not a bug**; it's the NEMA protected-left pattern, and manhattan (the
trusted reference) has the identical pattern.

---

## 7. Three ways a plan handles a conflict: protected, permissive, broken

Serving two conflicting movements green at once is **not** automatically a bug. Real signals —
and NEMA — allow three distinct relationships, and SUMO encodes each with a different state
character:

| Relationship | What happens | SUMO chars | Real-world example |
|---|---|---|---|
| **Protected** | The loser is held at **red**; the winner has full right of way. | winner `G`, foe `r` | Dedicated left-turn arrow, oncoming stopped. |
| **Permissive** | Both may go, but the minor movement **yields** (waits for a gap). Collision-free via SUMO's internal-junction *response* logic. | minor `g` (or `s` for right-on-red), major `G` | "Permissive left" filtering through oncoming; right-on-red. |
| **Broken** | Both get **priority** green and **neither yields** → SUMO simulates a collision / teleport. | two `G` foes, no yield | An actual signal-timing defect. |

Two consequences worth internalizing:

- **"Collision-free" is a low bar.** SUMO's yield logic makes almost any auto-generated plan
  collision-free — even a messy one — because permissive movements quietly wait. So "no
  collisions" does **not** imply "clean NEMA phasing." (This is why the audit in §8 needs a
  *style* check, not just a *safety* check.)
- **Protected vs permissive left is a genuine NEMA choice.** A protected-left plan gives lefts
  their own phase (`NL_SL`) with oncoming red; a permissive-left plan lets lefts *filter*
  (`g`) through gaps during the opposing through. Both are legitimate NEMA treatments that
  trade capacity against safety differently — and which one a network uses is visible directly
  in its state strings.

The grid4x4 bug sat *outside* all three categories: it stamped `s` ("green-must-stop", a
**right-turn** semantic) onto **through and left** lanes — a semantic mismatch, not any real
left-turn treatment. That's why it rendered as nonsensical purple lanes. PR #6 replaces it
with clean **protected** phasing (`G`/`r`), matching manhattan.

---

## 8. Do all the maps follow the NEMA convention? A cross-network audit

**Short answer:** the abstract action space is always NEMA-legal; the SUMO plans are all
collision-free and NEMA-consistent, but they differ in **which** NEMA-sanctioned left/right
treatments they encode. Two layers to check.

### 8.1 Layer A — the RL action space (`mplight.yml → signal_config`)

Only **five** networks define `phase_pairs`. Every pair, decoded, is a NEMA-legal
non-conflicting combination — but the encodings are **not uniform**:

| Network | Movement index | # phases | Order starts | Note |
|---|---|---|---|---|
| grid4x4  | 12-movement (incl. rights) | 8 | `NT_ST…` | same pairs as cologne3 |
| cologne3 | 12-movement | 8 (some 3–4 via `valid_acts`) | `NT_ST…` | real junctions expose fewer phases |
| hz4x4    | 12-movement | 8 | `ET_WT…` | same 8 phases, reordered |
| hz1x1    | **8-movement** (no rights) | 8 | `ET_WT…` | different index base |
| cologne1 | 8-movement | **4** | `ET_WT…` | through+left pairs only |

⚠️ **Consequence:** "phase 1" is *not* the same movement across maps (e.g. it's `NL_SL` on
grid4x4 but `NT_ST` on hz1x1). Any code reasoning about phases must read each map's own
`signal_config` — as `agent/mplight.py` does via `signal_config[map_name]` — and never
hard-code a phase→movement mapping. Networks *not* listed here (manhattan, arterial, atlanta,
ingolstadt) have no `phase_pairs`; they're driven by agents that read phases straight from the
world/net.

### 8.2 Layer B — the SUMO signal plans (`.net.xml`)

I audited all **15** `.net.xml` files with SUMO's own **foes** and **response** matrices
(method + validation in §8.3). **Every network is collision-free** (zero "broken" conflicts
from §7). They split into three *styles* by which optional chars they use — and, crucially,
**every map places those chars on the semantically correct movement**:

| Style | Chars used | Networks |
|---|---|---|
| **Pure protected** | `G` `r` `y` only | manhattan_28x7, **grid4x4 (PR #6)**, hangzhou_1x1 ×4, atlanta_1x5 |
| **+ right-on-red** | adds `s` on **right** turns | hangzhou_4x4 (gudang + hetero ×2) |
| **+ permissive left** | adds `g` on **left** turns / u-turns | arterial4x4, cologne1, cologne3 ×2, ingolstadt21 |

Verification that no other map repeats grid4x4's defect — **where each go-char actually lands**:

| Network | `s` chars land on | `g` chars land on | verdict |
|---|---|---|---|
| grid4x4 **original** (pre-fix) | right **+ through + left** | — | 🔴 the bug |
| grid4x4 **PR #6** / manhattan  | none | none | ✅ protected |
| hangzhou_4x4 (all)             | **right only** (1536) | none | ✅ right-on-red |
| arterial4x4                    | **right only** (384)  | **left only** (32) | ✅ correct |
| ingolstadt21 / cologne3        | none | **left / u-turn only** | ✅ permissive-left |

Only the *original* grid4x4 ever put a go-signal on through/left lanes. Every other map is
internally consistent: `s` sits on rights (right-on-red), `g` on lefts/u-turns
(permissive-left) — exactly as SUMO intends.

> **Minor smell:** `hangzhou_4x4_hetero_..._m.net.xml` has 192 `s` chars on *unmapped* link
> indices (the `_og` original does not) — a small link-index drift introduced by the hetero
> edit. Not conflict-causing, but worth a glance before relying on that variant.

### 8.3 How this was verified (so you can re-run it)

- **Conflict model:** for each phase, take every pair of `G` (priority-green) links; they are a
  *true* conflict only if SUMO's `foes` bit is set **and** neither yields via the `response`
  matrix (internal-junction wait). Right-turn links are excluded (always permitted).
- **Bit-order calibration:** foes/response strings are read right-to-left (char *n−1−j* = link
  *j*), calibrated so the trusted **manhattan_28x7** reference comes out clean.
- **Positive control:** the pre-fix `grid4x4` (git commit `ab3d7fe`) *must* flag as defective —
  and it does (5,616 permissive foe-overlaps; `s` on through/left) — proving the checker
  detects real dirt rather than rubber-stamping every network.

---

## 9. Why this matters for the realism benchmark project

Tie-ins to *"A Realism-Decomposed Benchmark for RL-TSC"*:

- **Reproduction gate (weeks 1–2).** Correct signal plans are a *prerequisite*: if grid4x4's
  phases are defective, any reproduced MPLight/CoLight number on it is untrustworthy. Fixing
  grid4x4 to valid NEMA phases (PR #6) is part of locking the pipeline before axes are added.
- **Pedestrian-phase axis.** In NEMA, pedestrian phases run **concurrent with the parallel
  through phase** (e.g. a N-S walk with Ø4/Ø8) and are gated by the same barrier. Adding a
  ped axis means extending the ring-barrier structure, not bolting on ad-hoc greens —
  understanding §4 is required to do it correctly and safely.
- **Action space is fixed across algorithms.** All 5 baselines choose among these same 8
  NEMA phases, so any ranking differences come from *policy*, not from different action
  spaces — important for the factorial study's internal validity.
- **Realism framing.** Real controllers add *actuation, coordination, min/max greens, and
  clearance (yellow+all-red) intervals* on top of this phase set. The benchmark's "idealized"
  setting collapses those; several realism axes (startup-lost-time, reaction time) are
  precisely about the timing layer that sits on top of the phase structure here.
- **Protected vs permissive lefts = a built-in realism variable (§7–§8).** Your target
  networks already differ here: grid4x4 uses *protected* lefts, while Cologne and Ingolstadt
  use *permissive* lefts (`g`, filtering through oncoming). That directly affects the
  **TTC-based safety surrogate** and any pedestrian-phase axis — permissive lefts create
  gap-acceptance conflicts that protected lefts don't. Hold this constant (or treat it as an
  explicit factor) so it isn't a hidden confound in the vulnerability analysis.

---

## 10. One-paragraph summary

An intersection has 12 movements (4 approaches × R/T/L); rights are always allowed. Movements
that cross are conflicts, and the **NEMA dual-ring/barrier** standard organizes non-conflicting
movements into 8 phases — odd = protected lefts, even = throughs — with a barrier keeping the
two streets apart. LibSignal's RL agents choose among exactly these 8 NEMA-legal phases
(`phase_pairs` in `mplight.yml`), so the action space is conflict-safe by construction — though
the *encoding* (8- vs 12-movement index, phase order) differs per map, so never hard-code a
phase→movement mapping. SUMO writes each phase as a per-lane state string (`G` priority-green,
`r` red, `g` permissive-yield, `s` right-on-red-stop). A conflict can be resolved three ways —
**protected** (foe red), **permissive** (foe yields), or **broken** (both priority, collide);
an audit of all 15 networks shows every one is collision-free and places `s`/`g` on the correct
movements, using protected, right-on-red, or permissive-left treatments. grid4x4's bug was `s`
on **through/left** lanes (purple, a right-turn semantic misapplied) — unique among the maps;
PR #6 fixes it by adopting manhattan_28x7's clean protected NEMA phases.

---

### Sources
- SUMO signal-state semantics: SUMO docs, *Simulation/Traffic_Lights* (state chars `G g r y u s o O`; `s` is the right-on-red "green-must-stop"; `g` is permissive-green that yields via the junction `response` matrix).
- NEMA dual-ring / ring-and-barrier: ITE *Traffic Signal Timing Manual*; FHWA *Signal Timing Manual* (ring-barrier, odd=left/even=through, Ø2/Ø6 coordinated; protected vs permissive left treatments).
- Repo ground truth: `configs/tsc/mplight.yml` (`signal_config.*.phase_pairs`), `data/raw_data/*/**.net.xml`, `data/raw_data/manhattan_28x7/*`, and grid4x4 git history (`ab3d7fe` = pre-fix).
- Cross-network audit: foes+response conflict check over all 15 `.net.xml`, calibrated on manhattan_28x7 with the pre-fix grid4x4 as positive control (§8.3).

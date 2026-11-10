# Project Plan

## Purpose

Build a configurable Battlegrounds simulation. The Week 1 engineering goal is
to make it easy to create and play specific game iterations: choose cards from
the JSON card library, select which cards and heroes are enabled, configure
game parameters, and enable the mechanics needed for that iteration without
rewriting the core engine.

Patch-perfect reproduction is not the primary goal. The spreadsheet and the
team's explicit content selections define what each iteration contains. When a
rule or card interaction is uncertain, record the uncertainty and current
decision, implement/test that decision, and revise it as the simulation is
built. Do not quietly substitute a different patch's data or assumptions.

The simulation engine must run without a GUI. A separate optional local GUI
supports human play and bot-match visualization; headless simulation stays
available for tests. Training and performance evaluation are out of scope for
Week 1.

## Week 1 goals

1. Create the foundation of a flexible, data-driven match engine.
2. Implement the first selected game iteration using the card library
   (`data/cards.json`, originally imported from the workbook) and an explicit
   team-curated roster.
3. Add an optional GUI for playing one seat and watching bot matches.

“Implement the game in its totality” means a complete match for the content and
rules enabled in a selected iteration: recruit actions work, combat resolves,
players can be eliminated, and placements are assigned. It does not require
every possible patch, card, or seasonal mechanic to be completed in the first
pass.

## Core design principles

- **Iteration profiles, not patch-specific engine forks:** use one shared engine
  with a data/profile bundle for each iteration. A profile identifies its card
  library file, enabled cards/heroes, rules and parameters, and supported
  mechanics.
- **One card library:** `data/cards.json` holds every card's identity, name,
  tribe/type, tier, base stats, printed text, and structured effects in one
  entry. Versions of a card from different patches are separate cards with
  separate IDs. Change a card by editing its library entry, and say so in the
  commit; there is no separate override layer.
- **Explicit content selection:** keep the enabled-card/hero roster in a
  profile or manifest so the team can include/exclude content per iteration
  without editing simulation logic.
- **Structured behavior, not guessed text parsing:** printed text is retained
  as reference. Effects that the team chooses to implement are encoded as
  explicit structured effect data/handlers. Flag unimplemented or ambiguous
  text rather than silently assigning behavior.
- **Configurable rules:** put intended-to-vary values (player count, economy,
  limits, pool counts, enabled tribes, damage, etc.) in validated configuration.
  Separate game mechanics from the selected values.
- **Visible uncertainty:** keep a short rules/decisions log with the question,
  current assumption, affected profile, and tests. Resolve questions when
  implementation reaches them; do not block unrelated engine work waiting for
  perfect historical certainty.
- **Reproducibility:** for a fixed iteration profile, configuration, seed, and
  action sequence, match setup and outcomes must replay deterministically.
- **GUI independence:** the GUI consumes engine state/events/legal actions; it
  does not own game rules or add delays to headless stepping.

## Implementation stages

Each stage ends with focused tests, a concise explanation of changed behavior
and remaining uncertainties. The user authorized autonomous implementation from Section 2F onward, then
clarified that Week 1 should stop before training and performance evaluation.
Keep this plan aligned with that Week 1 scope.

### Step-by-step collaboration and code approval

- The original collaboration preference was to review each implementation
  section before code changes. The user later explicitly authorized autonomous
  implementation from Section 2F onward and then the full roadmap; that
  authorization supersedes the per-section and stage approval gates below.
- Keep implementation incremental and tested. Communicate meaningful
  milestones, implementation assumptions, and genuinely blocking ambiguities;
  do not stop for routine approval between authorized sections.

### Stage 0 — Agree the first iteration contract (no code)

- Confirm the first workbook sheet/profile to implement (the current candidate
  is sheet `15.6` in `season -1 (beta).xlsx`).
- Inspect its columns and rows, then agree the first enabled-card roster. Use
  the sheet's own tribe/type values; the team's chosen roster—not a rule that
  requires a particular tribe count—determines what is in that iteration.
- Define the minimum profile/config fields needed to run a match and select
  content, while keeping additional values extensible.
- List the first mechanics needed by the chosen cards. Note ambiguous
  interactions as decisions to make during the relevant implementation stage.
- Identify the complete-match loop required for Week 1 and separate core
  blockers from later features.
- **Deliverable:** a small first-iteration checklist and data/profile sketch,
  with open questions marked. No code changes.
- **Approval gate:** show this contract and request approval before scaffolding
  or writing implementation code.

### Stage 1 — Project foundation and card-library-backed profiles

- Scaffold the package, dependencies, test runner, and developer commands.
- Load the card library into a normalized internal card model with each card's
  stats, printed text, and structured effects.
- Add a per-iteration manifest for selected/enabled cards and heroes plus
  configuration and supported mechanics. Keep team selections separate from
  the card library.
- Validate duplicate/missing IDs, invalid tiers/stats, missing references, and
  unknown structured effects with explicit errors.
- Test profile loading, content inclusion/exclusion, and invalid data.
- **Exit:** one iteration can be loaded and inspected reproducibly from its
  profile, with no card selection hidden in engine code.
- **Approval gate:** explain schema and loader behavior; request approval before
  implementing match/recruit behavior.

### Stage 2 — Match state and recruit-phase core

- Implement configurable player count (default eight), seeded lobby/content
  setup, per-player state, and match phases.
- Implement profile-driven gold, Tavern tiers/upgrades, shop size, a finite
  shared pool, and freeze behavior.
- Implement selected recruit actions: buy, sell, play, reroll, freeze/unfreeze,
  upgrade, board reorder, hero power, spells, triple/golden, and Discover as
  required by the first iteration profile.
- Expose legal actions as structured operations with parameters/targets for
  both bots and GUI; reject invalid actions explicitly.
- Add focused tests for rules, configuration changes, selected content,
  pool accounting, and action legality.
- **Exit:** players can finish a complete recruit phase using only configured
  content/rules.
- **Checkpoint:** report tests and open decisions before moving to combat.

### Stage 3 — Selected effects and combat

- Track permanent buffs separately from temporary combat state.
- Implement only the structured effects and keywords selected for the active
  iteration, using shared handlers where behavior is common across profiles.
- Implement attack order, random target selection, Taunt, combat triggers,
  shields, deaths, Deathrattles, Reborn, Windfury, Magnetic, Start of Combat,
  and other selected effects as required by the enabled roster.
- Make effect ordering explicit. If a rule is uncertain, log the chosen
  behavior and add a test so it can be changed deliberately later.
- Add scenario tests for each selected effect and important interactions.
- **Exit:** combat resolves reliably for the enabled roster; unsupported
  effects are identified rather than silently ignored.
- **Checkpoint:** report covered effects and uncertainties before match flow.

### Stage 4 — Matchmaking, elimination, placement, and replay

- Implement opponent pairing and configurable rematch avoidance.
- Implement odd-player ghost behavior and its board/source policy as an
  iteration rule.
- Apply the configured damage formula/caps, health/armor, elimination,
  simultaneous-death handling, final placement, and match termination.
- Add a random legal-action agent to exercise full matches.
- Record the iteration/profile, effective configuration, seed, and ordered
  actions. Replaying them must reproduce events and placements.
- **Exit:** complete seeded matches work for the first iteration; configuration
  and roster changes do not require engine forks.
- **Checkpoint:** demonstrate headless matches and replay before GUI work.

### Stage 5 — Optional play/watch GUI

- Build a lightweight local GUI as a separate client of engine snapshots,
  events, and legal actions.
- Support a human-controlled seat against bots and all-bot spectator mode.
- Show the information/actions needed to inspect and play the selected
  iteration; add pause/resume, single-step, playback speed, and replay.
- Keep UI imports/dependencies optional and wall-clock playback outside the
  engine.
- Verify that GUI and headless replays with the same profile, seed, and actions
  produce identical engine outcomes.
- **Exit:** human play and bot watching use the shared engine; headless matches
  remain usable without opening a window.
- **Week 1 scope:** stop after a new player can follow recruit actions, board
  order, automatic combat, and match outcomes in the GUI. No PPO, model
  training, batch evaluation, or competitive-performance claims.

## Suggested first iteration

- Start with the first selected workbook sheet (currently `15.6`) as an initial
  data source, not as a claim of complete historical patch reproduction.
- Workbook inspection found 80 card rows with non-empty tiers across tiers 1–6.
  The `Tribe` column contains Beast (18 rows), Demon (11), Mech (23), Murloc
  (10), and `-` (18 untyped rows). `Nightmare Amalgam` is untyped in the Tribe
  column and has `All` in the Type column. Use the workbook's own fields; do not
  count untyped cards as another tribe or invent cards to meet a tribe-count
  assumption.
- Curate a manageable explicit roster from that sheet, retaining exact base
  spreadsheet values and printed text.
- Let the team select the enabled tribes/cards for this iteration from actual
  sheet entries. Do not invent cards or require a fixed tribe count.
- Implement the effects needed by that selected roster; defer other sheet
  effects to later iterations.
- Use one documented rules/config profile. Add another profile when a concrete
  iteration needs it, and test that profile switching changes data/configuration
  without changing shared engine code.

### Decision (2026-11-06): JSON card library replaces runtime workbook loading

The workbook → selection → override → profile-effects pipeline (Sections
1A–1F below) was replaced to make card data easier to read and edit:

- `data/cards.json` holds all 1,092 cards from the workbook's 13 sheets, keeping
  the `sheet:row` IDs (e.g. `15.6:17`). Each entry has stats, printed text,
  `implemented`, and (for implemented cards) its structured `effects`. 87 cards
  are implemented: the 29-card initial roster plus 58 keyword-only cards
  (Taunt / Divine Shield / Poisonous). The rest are reference only and cannot be
  enabled.
- Profiles hold only `iteration_id`, `cards_file`, `rules`, and
  `enabled_card_ids`. `load_game_data(profile)` returns the profile and its
  enabled cards; effect bindings are derived from the library.
- `hsbg/data/workbook.py`, `hsbg/data/selection.py`, card overrides, source
  coordinates, and the `openpyxl` dependency were removed. The workbook stays in
  the repository as the original reference. Twenty seeded matches replayed
  identically before and after the change.

The section notes below describe the original implementation and are kept as
history.

### Foundation implementation progress and next approval

- **Section 1A — complete:** added Python project metadata/package markers and
  typed card-source, card-record, override, and iteration-profile models with
  focused tests.
- **Section 1B — complete:** added `hsbg/data/workbook.py`, a read-only loader
  for the workbook's nine-column card sheets. It maps source `⚔`/`✙` columns to
  model `attack`/`health`, retains raw row values and source coordinates,
  normalizes `-` tribe/type sentinels, and reports malformed workbook data.
  The workbook remains unchanged.
- **Section 1C — complete:** added `hsbg/data/profiles.py` with
  `load_profile(path) -> IterationProfile`. It validates the profile JSON
  object, required fields, strings, enabled-card ID list, and reasoned
  overrides (optional attack, health, tier, and text), and reports malformed
  JSON/schema/model data explicitly. The parser does not load the workbook,
  resolve IDs against card records, or apply overrides.
- **Section 1D — complete:** added `hsbg/data/selection.py` with
  `select_enabled_cards(profile, cards) -> tuple[CardRecord, ...]`. It rejects
  missing or duplicate supplied IDs and preserves the order in the profile.
  It does not apply overrides or select a team roster.
- **Section 1E — complete:** added `apply_profile_overrides(profile,
  selected_cards) -> tuple[CardRecord, ...]` to `hsbg/data/selection.py`. It
  immutably applies supported attack, health, tier, and text overrides while
  preserving selection order and original workbook values. Missing override
  targets and unsupported override fields are reported.
- **Initial roster — approved:** `data/profiles/initial-15.6.json` selects 29
  workbook cards. The team excluded repeat/trigger effects, persistent auras,
  random-target effects, history tracking, cross-player effects, and special
  attack behavior from this first roster. This selection is data in the
  profile, not hard-coded in Python.
- **Section 1F — complete:** the profile contains typed, explicit effect
  declarations for every selected card. Supported declarations cover passive
  keywords, fixed token summons, deterministic buffs, and hero damage on
  passive/Battlecry/Deathrattle triggers. The parser checks complete roster
  coverage and rejects unknown IDs, operations, triggers, and invalid
  parameters. These declarations are data only; they do not execute card
  behavior. `Nightmare Amalgam` has an explicit empty effect declaration
  because its `All` type is already retained in the imported source record.
- **Section 2A — complete:** added a frozen `GameRules` configuration model
  and required profile `rules` block for player count, initial gold
  progression/cap, buy/sell/reroll costs, Tavern tier bounds, and board/hand
  limits. Later configuration additions include starting health/armor,
  Tavern upgrade-cost progression, tier-based shop sizes, damage cap,
  rematch avoidance, and maximum match turns. The initial profile records
  configurable assumptions, not verified historical-patch rules.
- **Section 2B — complete:** added frozen state models for `ShopOffer`,
  zone-aware `MinionInstance`, `PlayerState`, and `MatchState`. Shop offers
  carry persistent modifiers separately from owned copies; each owned instance
  explicitly records whether it is in hand or on board. No state transitions,
  shop generation, actions, or card-effect execution were added.
- **Section 2C — complete:** added `buy_offer`, an immutable player-state
  transition for buying an existing shop offer into hand. It deducts configured
  gold, checks hand capacity and IDs, removes the selected offer, and carries
  persistent modifiers into the owned instance. It does not execute
  Battlecries, generate offers, or update a shared pool; it is not yet an
  integrated pool-aware purchase.
- **Section 2D — complete:** the profile now configures copies per enabled card
  by Tavern tier, defaulting to 16/15/13/11/9/7 for tiers 1–6. Added an
  immutable initial pool inventory with one entry per enabled card ID; there
  is no cap on the number of unique card definitions per tier. For the current
  29-card roster this initializes 382 copies total.
- **Section 2E — complete:** each pool entry now tracks configured total and
  available copies. Immutable `take(card_id)` and `return_copy(card_id)`
  operations preserve the original pool and reject unknown, exhausted, or
  over-returned copies. They are not yet connected to player actions.
- **Section 2F — complete:** `draw_shop_offers` draws from supplied eligible
  card IDs, weighted by each definition's available copies, with a caller-
  supplied seeded RNG. Each offer reserves a copy; when supply is short, the
  shop contains only the offers that could be drawn. Invalid or duplicate
  identifiers are rejected. Tavern eligibility, configured shop sizes, and
  player shop installation/refresh/freeze remain out of scope.
- **Section 2G — complete:** `reroll_shop` performs a pool-aware, immutable
  shop replacement. It checks and charges the configured reroll cost, returns
  current offer copies before drawing replacements, and clears the freeze flag.
  Requested slot IDs and eligible cards remain explicit inputs; this does not
  infer Tavern eligibility or shop sizes. An inconsistent shop/pool state or
  invalid draw request raises explicitly without changing the input states.
- **Section 2H — complete:** `play_minion` moves an existing hand instance
  onto the end of the board, preserving its persistent modifiers and enforcing
  the configured board limit. It is a state transition only; Battlecries and
  other declared effects are not executed.
- **Section 2I — complete:** `set_shop_frozen` explicitly freezes or unfreezes
  the current offers without charging gold or otherwise changing player state.
  A later turn-advance operation will determine when frozen offers persist.
- **Section 2J — complete:** `sell_minion` sells a board minion, applies the
  configured refund up to the gold cap, and returns a pool copy when
  `MinionInstance.pool_owned` is true. Non-pool tokens do not add copies to the
  pool. Invalid state transitions or pool accounting fail explicitly.
- **Section 2K — complete:** `reorder_board` accepts an explicit ordering of
  every current board instance and preserves hand order. Duplicate, missing,
  unknown, or invalid instance IDs are rejected; an empty board accepts only
  an empty order.
- **Section 2L — complete:** Tavern upgrades use an explicit per-tier cost
  schedule stored in each iteration's `GameRules` profile. The initial
  profile's baseline is 5/7/8/9/10 gold to reach tiers 2–6; these are
  configurable starting assumptions, not a claim of exact patch rules.
  `upgrade_tavern` charges the cost for the next tier and enforces the maximum
  tier. Turn-based cost discounts/reset behavior remains unimplemented.
- **Section 2M — complete:** `start_recruit_turn` sets gold from the configured
  starting amount and per-turn increase, capped at `max_gold`. A frozen shop
  persists for that refresh and is then unfrozen; otherwise old offers return
  to the pool and a fresh shop is drawn. Slot IDs and eligible card IDs remain
  inputs, and turn-based Tavern-upgrade discounts are not applied.
- **Section 2N — complete:** `create_initial_match` creates a turn-one recruit
  lobby with the configured player count, shared initial pool, starting
  resources/tiers, and sequentially dealt opening shops. Only cards at or
  below the starting Tavern tier are eligible. Starting health/armor and shop
  sizes come from the profile; offer IDs are generated deterministically.
- **Section 2O — complete:** `get_legal_recruit_actions` exposes stable,
  structured buy/play/sell/reroll/freeze/unfreeze/upgrade/reorder/end-turn
  choices derived from current gold, limits, and state. This catalog is
  informational; action execution must revalidate against current state.
- **Sections 2P–2Q — complete:** `GameEngine` validates and applies legal
  recruit actions, resolves selected Battlecries and combat, advances turns,
  handles elimination/placement/ghost pairing, and records replay actions.
  A replay also carries its complete immutable profile (including effective
  rules/effects) and selected card records; replay rejects changed inputs.
- **Stage 3 — complete for the selected roster:** the effect interpreter and
  seeded combat cover the roster's declared Battlecries, summons, buffs,
  Taunt, Divine Shield, Poisonous, and Deathrattle summons. Other mechanics
  remain unsupported until explicitly added to a profile.
- **Stage 4 — complete:** a random legal-action bot runs bounded complete
  matches, and action replay reproduces seeded state, combat, and placements.
  Rematch avoidance, odd-lobby ghost policy, damage assumptions, and match-turn
  limit are configurable. These choices have focused scenario tests.
- **Stage 5 — implemented:** `hsbg.play` offers human-versus-bot and spectator
  modes with pause, single-step, speed control, and replay. The GUI delegates
  decisions to the shared engine; headless dependencies remain independent
  of Tkinter.
- **Week 1 scope — implemented:** a readable lobby and player area show health,
  armor, gold, Tavern tier, shop/hand/board cards and stats; action buttons
  state their costs and effects; a plain-language event feed records bot/human
  actions and presents combat events one at a time at the selected playback
  speed. Removed PPO, training, and batch-evaluation modules and dependencies.
  Random bots remain as opponents and spectators.
- **Current verification:** after this scope/UI update, run the full test suite,
  Ruff, formatting, and mypy. Smoke-test the tutorial, recruit-action log,
  combat event playback, and GUI replay.

## Decisions to record as implementation reaches them

This is a living checklist, not a demand to resolve every historical question
before any coding:

- Which sheet/version and exact card/hero IDs are selected for each iteration.
- How the source workbook represents tribes, untyped minions, golden cards,
  stats, and effects; how selected cards are imported and overrides are stored.
- Historical accuracy of the initial profile's configurable starting
  health/armor, lobby size, shop offer rules, pool copies, player gold
  progression/cap, costs, Tavern tiers/shop sizes, and board/hand limits.
  The current values are explicit assumptions and are captured by tests.
- Golden minion pool accounting and Discover offer generation.
- Hero selection, hero powers (active, passive, start-of-game), costs/targets,
  and effect timing.
- Tavern spell availability, costs, targeting, and effects.
- Selected effect definitions and timing/order, including Battlecry,
  Deathrattle, Taunt, Divine Shield, Reborn, Windfury, Poisonous/Magnetic,
  Avenge or other effects only when included in a profile.
- Matchmaking/recent-opponent policy, ghost source/behavior, attack order,
  targeting, no-attack handling, damage formula/caps, ties, and simultaneous
  eliminations.
- Which values belong in generic config versus per-profile content/rule data.

For every unresolved item, record: question, current implementation choice,
profiles affected, and a test that captures the choice. Update the choice when
the team decides differently.

## Later work (not part of Week 1)

- If the team later chooses to resume learning experiments, training/evaluation
  can be proposed as a separate approved project stage.
- Consider seasonal systems (Anomalies, Trinkets, Quests, Buddies) as separately
  scoped, profile-selectable work.

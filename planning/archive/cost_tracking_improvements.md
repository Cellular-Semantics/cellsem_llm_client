# Cost Tracking Improvements Plan

## Scope
- Default cost estimation when tracking is enabled.
- Opt-out flag for auto cost estimation.
- Documentation updates for estimated vs actual costs.
- Rate updater script and weekly GitHub Action.
- Surface rate freshness date in cost metrics/reporting.

## Tasks
1) **Auto cost estimator default**
   - In `query_unified` (and wrappers), when `track_usage=True` and no `cost_calculator` is provided, auto-create `FallbackCostCalculator()` (load default rates) so `estimated_cost_usd` is populated by default.
   - Add an opt-out flag (e.g., `auto_cost=True`) to disable auto estimation.
   - Honor provided calculators; if `auto_cost=False`, skip estimation.

2) **Docs**
   - README: note default estimated costs (unless opted out) and link to cost tracking doc.
   - New/updated doc (`docs/cost_tracking.md`): per-request estimated costs (auto/explicit calculator, opt-out), key-level actual usage via `ApiCostTracker` with code snippets/caveats (provider delay, key-level totals), and guidance on estimates vs actuals.

3) **Rate updater + weekly Action**
   - Add `scripts/update_rates.py` to fetch latest OpenAI/Anthropic pricing and update the rate database used by `FallbackCostCalculator`.
   - Add a scheduled GitHub Action (weekly) to run the updater; assume no secrets needed. If rates change, prepare/flag a PR or failing status.

4) **Rate freshness in metrics**
   - Include rate database last-update date in cost tracking output/metrics so users see pricing freshness.

5) **Execution notes**
   - Implement code changes after this plan is approved.
   - Keep backward compatibility; new defaults are opt-out.

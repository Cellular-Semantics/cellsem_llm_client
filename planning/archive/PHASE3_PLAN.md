# Phase 3 Implementation Plan: Token Tracking & Cost Monitoring

## Overview
Implement **Priority 1** from the roadmap: comprehensive token tracking and cost monitoring with real-time API integration as the primary method and fallback cost calculation.

## Current State Analysis
- ✅ **Phase 2 Complete**: Core multi-provider functionality with comprehensive testing
- ✅ **Documentation**: Automated documentation system deployed
- 📍 **Current branch**: `feature/automated-documentation` (needs switching to new branch)
- 🎯 **Next Priority**: Token Tracking & Cost Monitoring (Week 1-2 in roadmap)

## Implementation Plan

### Step 1: Create New Feature Branch
- Switch to main branch and pull latest changes
- Create new branch: `feature/token-tracking-cost-monitoring`

### Step 2: TDD Implementation Structure
Following strict TDD workflow from CLAUDE.md:

1. **Write Tests First** (Red Phase)
   - Unit tests for `ApiCostTracker` class with OpenAI/Anthropic API mocking
   - Unit tests for `FallbackCostCalculator` with rate database functionality
   - Integration tests for real API cost tracking (local dev with real keys)
   - Unit tests for `UsageMetrics` data class with all token types

2. **Minimal Implementation** (Green Phase)
   - Implement basic `ApiCostTracker` to pass tests
   - Implement `FallbackCostCalculator` with static rate data
   - Implement `UsageMetrics` data class
   - Add enhanced `query_with_tracking()` method to `AgentConnection`

3. **Refactor & Enhance** (Refactor Phase)
   - Add automated rate database updates
   - Implement hybrid API/fallback cost strategy
   - Add comprehensive error handling and graceful degradation

### Step 3: Core Components to Implement

**New Module Structure:**
```python
src/cellsem_llm_client/tracking/
├── __init__.py
├── api_trackers.py      # Real-time API cost tracking
├── cost_calculator.py   # Fallback cost calculation
├── rate_database.py     # Automated rate updates
└── usage_metrics.py     # Enhanced usage data classes
```

**Key Classes:**
1. **`ApiCostTracker`**: Real-time cost tracking via provider APIs
   - OpenAI: `/v1/organization/usage` and `/v1/organization/costs`
   - Anthropic: `/v1/organizations/cost_report` and `/v1/organizations/usage_report`
   - 5-minute data availability per research

2. **`FallbackCostCalculator`**: Rate-based cost estimation
   - Weekly automated rate updates from provider documentation
   - Source tracking with access dates
   - Graceful degradation when API tracking unavailable

3. **`UsageMetrics`**: Enhanced usage data structure
   - Support for all token types: input, output, cached, thinking tokens
   - Actual vs estimated costs with source attribution
   - Provider and model metadata with timestamps

### Step 4: Enhanced Agent Integration
- Add `query_with_tracking()` method to `AgentConnection` base class
- Return tuple of `(response: str, usage: UsageMetrics)`
- Maintain backward compatibility with existing `query()` method
- Comprehensive error handling for API tracking failures

### Step 5: Testing Strategy
**Unit Tests:**
- Mock all external API calls (OpenAI/Anthropic cost APIs)
- Test calculation accuracy with known rate scenarios
- Test error handling and fallback mechanisms

**Integration Tests:**
- **Local Dev**: Real API calls to OpenAI/Anthropic cost endpoints
- **CI**: Controlled mocks via `USE_MOCKS=true`
- Test actual cost tracking accuracy vs provider billing

### Step 6: Quality Gates
- >85% test coverage requirement
- Type safety with MyPy compliance
- Real API integration test validation
- Performance overhead <10% for basic queries

## MVP Success Criteria
1. **>95% accuracy** vs actual API billing for cost tracking
2. **Real-time costs within 5%** of provider APIs
3. **Graceful fallback** when API tracking unavailable
4. **Comprehensive test coverage** with real API validation
5. **Type-safe implementation** with proper error handling

## Risks & Mitigation
1. **API Rate Limits**: Cost tracking APIs may have different limits
   - *Mitigation*: Implement caching, batch requests, graceful degradation

2. **Provider API Changes**: Cost API endpoints could change
   - *Mitigation*: Comprehensive error handling, fallback to rate calculation

3. **Token Type Complexity**: Different providers have different token categories
   - *Mitigation*: Unified `UsageMetrics` class with optional fields

## Branch Strategy
- All work in `feature/token-tracking-cost-monitoring` branch
- Frequent commits following TDD red-green-refactor cycle
- Regular code quality checks with ruff/mypy before commits
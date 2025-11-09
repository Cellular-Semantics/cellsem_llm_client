# ROADMAP.md - CellSem LLM Client Phase 3

## Executive Summary
Phase 3 transforms our TDD-based foundation into a production-ready LLM client with advanced features: **token tracking**, **cost monitoring**, **schema compliance**, **file attachments**, and **agentic model recommendations**.

## Research Findings

### API-Based Cost Tracking (✅ Available!)
**OpenAI**: Has comprehensive Usage & Costs APIs
- `/v1/organization/usage/{endpoint}` - Real-time token tracking
- `/v1/organization/costs` - Daily spend breakdown
- Granular filtering by API key, project, user, model
- Data available within 5 minutes of API calls

**Anthropic**: Full cost tracking capabilities
- `/v1/organizations/cost_report` - Service-level costs in USD
- `/v1/organizations/usage_report/messages` - Token consumption by model/workspace
- Includes new "thinking tokens" for Claude 4.1 series
- Data typically available within 5 minutes

**Conclusion**: Both major providers offer real-time cost APIs - we can implement direct API tracking as primary method!

### Schema Support Status
**OpenAI**: Native structured outputs with `strict: true`
- Full JSON schema enforcement via API
- Pydantic/Zod SDK integration
- Function calling with guaranteed schema adherence

**Anthropic**: Tool use pattern for structured outputs
- Use tools with JSON schema input specifications
- Force tool usage with `tool_choice` parameter
- Reliable but requires tool wrapper approach

**LiteLLM**: Abstracts both approaches seamlessly

### File Attachment Capabilities
**LiteLLM Support**: Comprehensive multi-provider file handling
- PDF, images, documents via `/chat/completions`
- Provider Files API endpoints in OpenAI format
- Unified vision models support across providers
- Base64 and URL-based file inputs
- Model capability checking with `litellm.supports_pdf_input()`

## Phase 3 Implementation Plan

### Priority 1: Token Tracking & Cost Monitoring 🔥
**(✅ DONE & Available!)*

**Core Components:**
```python
# Real-time API cost tracking (primary method)
class ApiCostTracker:
    def get_openai_usage() -> UsageReport
    def get_anthropic_usage() -> UsageReport
    def get_real_time_costs() -> CostBreakdown

# Fallback rate calculator with source tracking
class FallbackCostCalculator:
    def update_rate_database() -> RateUpdateReport  # Weekly automated updates
    def calculate_estimated_cost() -> EstimatedCost
    def get_rate_source_info() -> RateSourceInfo    # Track source + access date

# Enhanced usage metrics from API responses
class UsageMetrics:
    input_tokens: int
    output_tokens: int
    cached_tokens: int | None = None
    thinking_tokens: int | None = None  # Anthropic Claude 4.1+
    actual_cost_usd: float | None = None  # From API when available
    estimated_cost_usd: float | None = None  # Fallback calculation
    cost_source: Literal["api", "estimated"]
    provider: str
    model: str
    timestamp: datetime
```

**Implementation Strategy:**
1. **Primary**: Direct API cost tracking (both providers support this!)
2. **Fallback**: Rate database with automated updates and source attribution
3. **Hybrid**: Use API data where available, estimated costs as backup

### Priority 2: JSON Schema Compliance 🔥
**(✅ DONE & Available!)**

**Provider-Specific Implementation:**
```python
# Unified schema interface
class SchemaValidator:
    def validate_and_retry(response: str, schema: Type[BaseModel]) -> BaseModel
    def get_provider_schema_support(provider: str) -> SchemaCapabilities

# Enhanced agent methods
class LiteLLMAgent(AgentConnection):
    def query_with_schema(
        self,
        message: str,
        schema: Type[BaseModel],
        max_retries: int = 3,
        **kwargs
    ) -> BaseModel:
        # OpenAI: Use native structured outputs with strict=true
        # Anthropic: Use tool use pattern with schema enforcement
        # LiteLLM: Abstract the differences
```

**Two-Tier Approach:**
1. **Native API Schema Passing** (where supported)
   - OpenAI: `response_format` with `strict=true`
   - Anthropic: Tool use with `tool_choice` enforcement
2. **Post-Processing Validation** (universal fallback)
   - Pydantic validation with retry logic
   - Error-specific retry strategies

### Priority 3: File Attachment Support 🔥
**TODO**

**LiteLLM Integration:**
```python
class FileProcessor:
    def validate_file_capabilities(model: str, file_type: str) -> bool
    def encode_file_for_provider(file: Path, provider: str) -> FileEncoding
    def get_file_size_limits(provider: str, model: str) -> FileLimits

# Enhanced query methods
def query_with_files(
    self,
    message: str,
    files: list[Path | str | bytes],
    system_message: str | None = None,
    **kwargs
) -> str:
    # Use LiteLLM's unified file API
    # Support PDF, images, documents
    # Provider-specific capability checking
```

**Supported File Types:**
- **Images**: PNG, JPEG, WebP (all providers)
- **Documents**: PDF (most providers via LiteLLM)
- **Other**: Provider-specific extensions via LiteLLM abstraction

### Priority 4: Agentic Model Recommendations 🚀
**TODO**

**AI-Powered Advisory System:**
```python
class ModelAdvisor:
    def recommend_model_for_task(
        task_description: str,
        budget_per_request: float | None = None,
        response_time_requirement: str | None = None
    ) -> ModelRecommendation

    def estimate_token_requirements(
        input_text: str,
        expected_output_length: str,
        schema_complexity: str | None = None
    ) -> TokenEstimate

    def assess_task_complexity(prompt: str) -> ComplexityAssessment
    def get_model_capabilities_matrix() -> CapabilitiesMatrix
```

**Decision Factors:**
- Task complexity analysis (via meta-model)
- Cost per token considerations
- Required capabilities (vision, function calling, etc.)
- Performance benchmarks and latency requirements

## Architecture Evolution

### Enhanced Agent Base Class
```python
class AgentConnection(ABC):
    # Core methods (existing)
    @abstractmethod
    def query(self, message: str, **kwargs) -> str: pass

    # New tracking-enabled methods
    @abstractmethod
    def query_with_tracking(self, message: str, **kwargs) -> tuple[str, UsageMetrics]: pass

    # Schema and file support
    def query_with_schema(self, message: str, schema: Type[BaseModel], **kwargs) -> BaseModel: pass
    def query_with_files(self, message: str, files: list[Path], **kwargs) -> str: pass

    # Combined advanced query
    def query_advanced(
        self,
        message: str,
        schema: Type[BaseModel] | None = None,
        files: list[Path] | None = None,
        track_usage: bool = True,
        **kwargs
    ) -> AdvancedResponse: pass
```

### Project Structure Updates
```
src/cellsem_llm_client/
├── agents/                    # Existing
├── tracking/                  # NEW: Usage & cost tracking
│   ├── api_trackers.py       # Real-time API cost tracking
│   ├── cost_calculator.py    # Fallback cost calculation
│   └── rate_database.py     # Automated rate updates
├── schema/                    # Enhanced
│   ├── validators.py         # Cross-provider schema validation
│   └── adapters.py          # Provider-specific schema handling
├── files/                     # NEW: File handling
│   ├── processors.py         # File validation and encoding
│   └── capabilities.py      # Model file capability detection
├── advisors/                  # NEW: Model recommendations
│   ├── model_advisor.py      # Task-based model recommendations
│   ├── complexity_analyzer.py # AI-powered task analysis
│   └── benchmarks.py        # Performance and cost benchmarks
└── utils/                     # Enhanced configuration
```

## Success Metrics & Quality Gates

### MVP Completion Criteria
- [ ] **Token Tracking**: >95% accuracy vs actual API billing
- [ ] **Cost Monitoring**: Real-time costs within 5% of provider APIs
- [ ] **Schema Validation**: >90% success rate for well-formed schemas
- [ ] **File Attachments**: Support for images, PDFs across major providers
- [ ] **Model Recommendations**: Measurable improvement in cost/performance

### Quality Standards
- [ ] All features maintain >85% test coverage
- [ ] Integration tests with real provider APIs
- [ ] Type safety (MyPy clean)
- [ ] Performance impact <10% overhead for basic queries
- [ ] Comprehensive documentation with working examples

## Risk Mitigation

### High-Priority Risks
1. **API Rate Limit Impacts**: Cost tracking APIs may have different limits
   - *Mitigation*: Implement caching, batch requests, graceful degradation

2. **Provider Schema Differences**: Anthropic vs OpenAI schema handling varies
   - *Mitigation*: Provider-specific adapters with unified interface

3. **File Upload Complexity**: Each provider has different capabilities/limits
   - *Mitigation*: Use LiteLLM abstraction, clear capability detection

### Implementation Timeline
- **Week 1-2**: Token tracking with real-time API integration
- **Week 3-4**: Schema compliance (native + fallback validation)
- **Week 5-6**: File attachment support via LiteLLM
- **Week 7-8**: Model advisory system with complexity analysis
- **Week 9**: Integration testing, performance optimization, documentation

This roadmap prioritizes features with the highest immediate value (cost tracking, schema validation) while building toward the advanced AI-powered recommendations that differentiate our client from basic LiteLLM usage.

# Phase 3 Priority 2: JSON Schema Compliance with Provider Integration & Build System

## Current State Analysis
✅ **Priority 1 (Token Tracking & Cost Monitoring)** - COMPLETED
- `ApiCostTracker`, `FallbackCostCalculator`, `UsageMetrics` all implemented
- Ready for JSON Schema-first implementation with native provider support

## Implementation Plan: Schema Compliance with Native Provider Integration

### Phase 2A: Schema Infrastructure & Provider Adapters (Week 1)

**Core Provider Schema Compliance**:
```python
# Enhanced agent methods with native provider support
def query_with_schema(
    self,
    message: str,
    schema: str | dict | Type[BaseModel] | URL,
    max_retries: int = 3,
    cost_calculator: Optional[FallbackCostCalculator] = None,
    **kwargs
) -> tuple[BaseModel, UsageMetrics]:
    """
    Native provider schema enforcement:
    - OpenAI: Uses response_format with strict=True
    - Anthropic: Uses tool_choice with JSON schema tools
    - Others: Post-processing validation with retry logic
    """
```

**Provider-Specific Implementation**:

1. **OpenAI Integration** (`schema/adapters.py`):
   ```python
   class OpenAISchemaAdapter:
       def apply_native_schema(self, messages, schema_dict):
           # Use OpenAI's native structured outputs
           return completion(
               messages=messages,
               response_format={
                   "type": "json_schema",
                   "json_schema": {
                       "name": "response",
                       "schema": schema_dict,
                       "strict": True  # Enforce exact compliance
                   }
               }
           )
   ```

2. **Anthropic Integration** (`schema/adapters.py`):
   ```python
   class AnthropicSchemaAdapter:
       def apply_native_schema(self, messages, schema_dict):
           # Convert schema to tool definition
           tool = {
               "name": "structured_response",
               "description": "Provide structured response",
               "input_schema": schema_dict
           }
           return completion(
               messages=messages,
               tools=[tool],
               tool_choice={"type": "tool", "name": "structured_response"}
           )
   ```

3. **Universal Fallback** (`schema/validators.py`):
   ```python
   class SchemaValidator:
       def validate_with_retry(self, response_text, schema, max_retries=3):
           # Pydantic validation with intelligent retry
           for attempt in range(max_retries):
               try:
                   return parse_obj_as(target_model, response_text)
               except ValidationError as e:
                   # Retry with error-specific prompting
                   response_text = self._retry_with_validation_error(e)
   ```

### Phase 2B: Build Integration & Schema Management (Week 2)

**Build-Integrated Schema System** (Your requirements):
```
my_project/
├── schemas/                          # Convention-based discovery
│   ├── my_task.json                 # Local schemas
│   ├── remote_refs.json             # URL-based schemas
│   └── config.yaml                  # Optional configuration
├── .cellsem/                        # Generated (gitignored)
│   ├── generated_models/            # Auto-generated Pydantic
│   └── schema_cache/                # URL schema cache
└── pyproject.toml                   # Build integration
```

**Seamless Provider Integration Usage**:
```python
# User code - provider-agnostic but optimally integrated
from cellsem_llm_client import create_openai_agent
from cellsem_llm_client.schema import get_model

MyTask = get_model("my_task")  # From schemas/my_task.json

# OpenAI: Uses native structured outputs automatically
openai_agent = create_openai_agent()
result, usage = openai_agent.query_with_schema(
    "Complete this task: ...",
    MyTask  # Native schema enforcement via response_format
)

# Anthropic: Uses tool choice pattern automatically
anthropic_agent = create_anthropic_agent()
result, usage = anthropic_agent.query_with_schema(
    "Complete this task: ...",
    MyTask  # Native schema enforcement via tools
)

# Other providers: Fallback validation with retry
other_agent = create_litellm_agent(model="some-other-model")
result, usage = other_agent.query_with_schema(
    "Complete this task: ...",
    MyTask  # Post-processing validation
)
```

### Two-Tier Schema Compliance Strategy

**Tier 1: Native Provider Integration** (Primary):
- **OpenAI**: Direct `response_format` with `strict=true` - guaranteed compliance
- **Anthropic**: `tool_choice` enforcement - forces tool use with schema
- **LiteLLM Detection**: Auto-detect provider capabilities and route accordingly

**Tier 2: Universal Fallback** (Secondary):
- **Validation**: Pydantic parsing with detailed error reporting
- **Retry Logic**: Intelligent retry with validation error context
- **Error Recovery**: Progressive fallback strategies for complex schemas

### Technical Architecture

**Provider Auto-Detection**:
```python
class SchemaAdapterFactory:
    def get_adapter(self, model: str, provider: str):
        if provider == "openai":
            return OpenAISchemaAdapter()  # Native structured outputs
        elif provider == "anthropic":
            return AnthropicSchemaAdapter()  # Tool choice pattern
        else:
            return FallbackSchemaAdapter()  # Validation + retry
```

**Schema Processing Pipeline**:
1. **Schema Resolution**: Local file, URL, or runtime dict
2. **Provider Detection**: Determine optimal compliance method
3. **Native Integration**: Use provider-specific schema passing
4. **Response Validation**: Parse and validate against Pydantic model
5. **Error Handling**: Retry with context if validation fails

### Build Integration Features

**Auto-Generation Pipeline**:
```toml
[tool.cellsem-schema]
input_directories = ["schemas/"]
remote_schemas = true                    # URL resolution support
output_directory = ".cellsem/generated_models/"
type_checking = "warn-exclude"          # Avoid IDE conflicts
provider_optimization = true            # Generate provider-specific hints
```

**URL Schema Support**:
```json
// schemas/api_response.json
{
  "$ref": "https://api.openai.com/schemas/function_call.json",
  "allOf": [{"$ref": "#/$defs/custom_fields"}]
}
```

### Implementation Timeline

#### Week 1: Core Schema Infrastructure
**Days 1-2**: Schema Foundation
- [ ] Schema resolution and loading system
- [ ] Basic Pydantic model integration
- [ ] JSON Schema validation utilities

**Days 3-4**: Provider Adapters
- [ ] OpenAI `response_format` integration
- [ ] Anthropic tool choice pattern
- [ ] Provider auto-detection logic

**Days 5-7**: Agent Integration
- [ ] `query_with_schema()` method implementation
- [ ] Integration with existing `query_with_tracking()`
- [ ] Error handling and retry logic

#### Week 2: Build Integration & Advanced Features
**Days 8-10**: Schema Management
- [ ] Build system integration (setuptools/uv hooks)
- [ ] Auto-generation pipeline
- [ ] Convention-based schema discovery

**Days 11-12**: URL Schema Support
- [ ] Remote schema resolution
- [ ] Caching and update mechanisms
- [ ] Network error handling

**Days 13-14**: Developer Experience
- [ ] Type checking integration
- [ ] CLI tools for schema management
- [ ] Documentation and examples

### Success Criteria
- [ ] **Native Provider Support**: OpenAI structured outputs + Anthropic tools working
- [ ] **>90% Schema Compliance**: With provider native methods where available
- [ ] **Universal Fallback**: Validation + retry for all other providers
- [ ] **Build Integration**: Seamless pip/uv installation with schema generation
- [ ] **URL Schema Resolution**: Remote schema fetching with caching
- [ ] **Type Safety**: Generated models with configurable type checking
- [ ] **>85% Test Coverage**: Including provider-specific integration tests

### Risk Mitigation
1. **Provider API Changes**: Use LiteLLM abstraction + fallback validation
2. **Schema Complexity**: Implement progressive validation strategies
3. **Build Integration**: Optional generation with graceful fallbacks
4. **Performance Impact**: Benchmark and optimize validation overhead
5. **Type Checking Conflicts**: Configurable inclusion/exclusion strategies

This plan delivers the core value: **native provider schema compliance** while providing seamless build integration and developer experience features.
# Planning Documents

This directory contains all planning and strategy documents for the CellSem LLM Client project.

## Documents

### Core Planning
- **[ROADMAP.md](ROADMAP.md)** - Phase 3 roadmap with advanced features (token tracking, schema compliance, file attachments, AI recommendations)
- **[PHASE3_PLAN.md](PHASE3_PLAN.md)** - Detailed implementation plan for Phase 3 features
- **[project_setup_plan.md](project_setup_plan.md)** - Original project setup and architecture planning

### Testing & Validation
- **[dashboard-testing-plan.md](dashboard-testing-plan.md)** - Comprehensive testing strategy for cost tracking validation against OpenAI and Anthropic dashboards

## Quick Reference

### Current Status
- ✅ **Phase 1**: Basic LiteLLM integration
- ✅ **Phase 2**: Multi-provider support with comprehensive testing
- 🔄 **Phase 3**: Advanced features in development

### Phase 3 Priorities
1. 🔥 **Token Tracking & Cost Monitoring** (Primary focus)
2. 🔥 **JSON Schema Compliance**
3. 📎 **File Attachment Support**
4. 🤖 **AI-Powered Model Recommendations**

### Key Implementation Notes
- All development follows strict TDD (Test-Driven Development)
- Real API integration for local testing, mocks for CI
- >85% test coverage required
- Type safety with MyPy enforcement

## Related Documents
- [Contributing Guidelines](../docs/contributing.md)
- [Development Rules](../CLAUDE.md)
- [API Documentation](../docs/)
# Architecture Map

## Service Dependencies

```mermaid
flowchart TD
    app_src_api_main_py --> src
    app_src_data_uv_vis_dataset_py --> src
    app_src_drivers_uv_vis_py --> src
    app_src_lab_campaign_py --> configs
    app_src_lab_campaign_py --> src
    app_src_reasoning_dual_agent_py --> src
    app_src_services_db_py --> src
    app_src_services_storage_py --> src
    app_tests_test_health_py --> src
    app_tests_test_reasoning_smoke_py --> src
    app_tests_unit_test_uv_vis_driver_py --> src
    apps_api_main_py --> services
    configs_data_schema_py --> pint
    labloop_tests_test_safety_guard_py --> labloop
    labloop_tests_test_scheduler_py --> labloop
    labloop_tests_test_sim_adapter_py --> labloop
    pilotkit_tests_test_feedback_iteration_py --> pilotkit
    pilotkit_tests_test_metrics_py --> pilotkit
    pilotkit_tests_test_selection_py --> pilotkit
    pilotkit_tests_test_stats_py --> pilotkit
    scripts_bootstrap_py --> configs
    scripts_bootstrap_py --> src
    scripts_run_uv_vis_campaign_py --> src
    scripts_train_ppo_py --> src
    scripts_train_ppo_expert_py --> src
    scripts_train_rl_agent_py --> src
    scripts_validate_rl_system_py --> src
    scripts_validate_stochastic_py --> scripts
    scripts_validate_stochastic_py --> src
    services_agents_orchestrator_py --> services
    services_evals_report_py --> services
    services_evals_runner_py --> services
    services_llm_guardrails_py --> services
    services_rag_index_py --> services
    services_rag_pipeline_py --> services
    services_telemetry_dash_data_py --> services
    services_telemetry_store_py --> services
    src_connectors_simulators_py --> configs
    src_experiment_os_core_py --> configs
    src_experiment_os_core_py --> src
    src_reasoning_agentic_optimizer_py --> src
    src_reasoning_eig_optimizer_py --> configs
    src_safety_gateway_py --> configs
    synthloop_tests_test_orchestrator_py --> synthloop
    tests_test_safety_gateway_py --> configs
    tests_test_safety_gateway_py --> src
```

## Run Targets

- `make run.api` — start the FastAPI RAG service on :8000.
- `make run.web` — launch the marketing/demo shell.
- `make demo` — boot the Next.js demos workspace.
- `make graph` — refresh this dependency map.
- `make audit` — regenerate audit findings JSON.

## Data Model

```mermaid
erDiagram
    ExperimentRun {
        string id PK
        string query
        json context
        json flash_response
        json pro_response
        float flash_latency_ms
        float pro_latency_ms
        datetime created_at
        string user_id
    }
    InstrumentRun {
        string id PK
        string instrument_id
        string sample_id
        string campaign_id
        string status
        json metadata_json
        string notes
        datetime created_at
        datetime updated_at
    }
    ExperimentRun ||--o{ InstrumentRun : logs
```

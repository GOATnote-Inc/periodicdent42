# 🔧 Production Hardening Tasks - Priority Queue

**Date**: October 1, 2025  
**Context**: Pre-Phase 2A quick wins to improve production reliability  
**Estimated Total Time**: 2-3 days (before starting Phase 2A Sprint 1)

---

## 🎯 Recommended Execution Order

### 🔥 **CRITICAL - Do First** (4-6 hours)

#### 1. Health Check Alignment ⚡ (1 hour)
**Priority**: CRITICAL - Blocks production deployments  
**Effort**: 1 hour  
**Impact**: HIGH - Prevents false alarms in Cloud Run

**Problem**: README says `/healthz`, code exposes `/health` → health probes fail

**Implementation**:
```python
# app/src/api/main.py

@app.get("/health", tags=["system"])
@app.get("/healthz", tags=["system"])  # Add alias
async def health_check():
    """
    Health check endpoint for Cloud Run and load balancers.
    
    Available at both /health and /healthz for compatibility.
    """
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now(UTC).isoformat(),
        "service": "ard-intelligence",
        "version": "1.0.0"
    }
    
    # Check Vertex AI initialization
    if agent is None:
        health_status["status"] = "degraded"
        health_status["warnings"] = ["Vertex AI not initialized"]
        return JSONResponse(
            status_code=503,
            content=health_status
        )
    
    return health_status
```

**Tests**:
```python
# app/tests/test_health.py

def test_health_endpoint(client):
    """Test /health endpoint."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_healthz_alias(client):
    """Test /healthz alias works identically."""
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

def test_health_endpoints_identical(client):
    """Verify both endpoints return identical responses."""
    health_response = client.get("/health").json()
    healthz_response = client.get("/healthz").json()
    assert health_response == healthz_response
```

**Verification**:
```bash
# Local test
curl http://localhost:8080/health
curl http://localhost:8080/healthz

# After deployment
curl https://your-service.run.app/health
curl https://your-service.run.app/healthz
```

---

#### 2. Vertex Initialization Error Surfacing ⚡ (2 hours)
**Priority**: HIGH - Critical for ops visibility  
**Effort**: 2 hours  
**Impact**: HIGH - Enables recovery without restarts

**Problem**: Vertex init failures swallowed → app runs with `agent=None` → all requests 503 with no diagnostics

**Implementation**:
```python
# app/src/services/vertex.py

class VertexAIStatus:
    """Track Vertex AI initialization state."""
    def __init__(self):
        self.initialized = False
        self.last_error: Optional[str] = None
        self.last_attempt: Optional[datetime] = None
        self.retry_count = 0
    
    def mark_success(self):
        self.initialized = True
        self.last_error = None
        self.retry_count = 0
    
    def mark_failure(self, error: Exception):
        self.initialized = False
        self.last_error = str(error)
        self.last_attempt = datetime.now(UTC)
        self.retry_count += 1

# Global status
vertex_status = VertexAIStatus()

def init_vertex(
    project_id: str,
    location: str = "us-central1",
    flash_model: str = "gemini-2.5-flash",
    pro_model: str = "gemini-2.5-pro"
) -> Optional[DualModelAgent]:
    """
    Initialize Vertex AI with error tracking.
    """
    try:
        vertexai.init(project=project_id, location=location)
        agent = DualModelAgent(
            flash_model_name=flash_model,
            pro_model_name=pro_model
        )
        vertex_status.mark_success()
        logger.info("vertex_initialized", project=project_id, location=location)
        return agent
    except Exception as e:
        vertex_status.mark_failure(e)
        logger.error(
            "vertex_initialization_failed",
            error=str(e),
            retry_count=vertex_status.retry_count
        )
        return None

# Add retry helper
async def retry_vertex_init() -> bool:
    """Retry Vertex AI initialization."""
    from src.utils.settings import settings
    
    agent = init_vertex(
        project_id=settings.PROJECT_ID,
        location=settings.LOCATION,
        flash_model=settings.GEMINI_FLASH_MODEL,
        pro_model=settings.GEMINI_PRO_MODEL
    )
    
    if agent:
        # Update global agent reference
        import src.api.main as main_module
        main_module.agent = agent
        return True
    return False
```

```python
# app/src/api/main.py

@app.get("/health", tags=["system"])
@app.get("/healthz", tags=["system"])
async def health_check():
    """Health check with Vertex AI status."""
    from src.services.vertex import vertex_status
    
    health_status = {
        "status": "healthy",
        "timestamp": datetime.now(UTC).isoformat(),
        "service": "ard-intelligence",
        "version": "1.0.0",
        "components": {
            "vertex_ai": {
                "initialized": vertex_status.initialized,
                "last_error": vertex_status.last_error,
                "last_attempt": vertex_status.last_attempt.isoformat() if vertex_status.last_attempt else None,
                "retry_count": vertex_status.retry_count
            }
        }
    }
    
    if not vertex_status.initialized:
        health_status["status"] = "degraded"
        return JSONResponse(status_code=503, content=health_status)
    
    return health_status

@app.post("/admin/vertex/retry", tags=["admin"])
async def retry_vertex_initialization():
    """
    Admin endpoint to retry Vertex AI initialization.
    
    Requires API key authentication (already enforced by middleware).
    """
    from src.services.vertex import retry_vertex_init
    
    success = await retry_vertex_init()
    
    if success:
        return {"status": "success", "message": "Vertex AI initialized"}
    else:
        from src.services.vertex import vertex_status
        return JSONResponse(
            status_code=503,
            content={
                "status": "failed",
                "error": vertex_status.last_error,
                "retry_count": vertex_status.retry_count
            }
        )
```

**Tests**:
```python
# app/tests/test_vertex_init.py

def test_vertex_init_failure_tracking():
    """Test that init failures are tracked."""
    from src.services.vertex import init_vertex, vertex_status
    
    # Simulate failure
    agent = init_vertex(project_id="invalid-project")
    
    assert agent is None
    assert not vertex_status.initialized
    assert vertex_status.last_error is not None
    assert vertex_status.retry_count > 0

def test_health_check_shows_vertex_error(client):
    """Health check exposes Vertex initialization errors."""
    # With mock that forces init failure
    response = client.get("/health")
    
    assert response.status_code == 503
    data = response.json()
    assert data["status"] == "degraded"
    assert "vertex_ai" in data["components"]
    assert not data["components"]["vertex_ai"]["initialized"]
```

**Benefits**:
- ✅ Operators see **why** Vertex init failed
- ✅ Can retry without pod restart
- ✅ Health checks provide actionable diagnostics

---

### 🔧 **HIGH PRIORITY - Do Second** (2-4 hours)

#### 3. Local Storage Backend ⚡ (2-3 hours)
**Priority**: HIGH - Enables local testing  
**Effort**: 2-3 hours  
**Impact**: MEDIUM - Improves developer experience

**Problem**: No GCS credentials → storage endpoints always 503 → can't test locally

**Implementation**:
```python
# app/src/services/storage.py

class LocalStorageBackend:
    """Filesystem-based storage for local development."""
    
    def __init__(self, base_path: str = "./local_storage"):
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"LocalStorageBackend initialized at {self.base_path}")
    
    def store_experiment_result(
        self,
        experiment_id: str,
        result: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """Store experiment result to local filesystem."""
        experiment_dir = self.base_path / experiment_id
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        # Store result
        result_path = experiment_dir / "result.json"
        with open(result_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        # Store metadata
        if metadata:
            metadata_path = experiment_dir / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"Stored experiment {experiment_id} locally")
        return f"file://{result_path.absolute()}"
    
    def list_experiments(
        self,
        limit: int = 100,
        offset: int = 0
    ) -> List[Dict[str, Any]]:
        """List experiments from local storage."""
        experiments = []
        
        for exp_dir in sorted(self.base_path.iterdir(), reverse=True):
            if not exp_dir.is_dir():
                continue
            
            result_path = exp_dir / "result.json"
            metadata_path = exp_dir / "metadata.json"
            
            if result_path.exists():
                with open(result_path) as f:
                    result = json.load(f)
                
                metadata = {}
                if metadata_path.exists():
                    with open(metadata_path) as f:
                        metadata = json.load(f)
                
                experiments.append({
                    "experiment_id": exp_dir.name,
                    "result": result,
                    "metadata": metadata,
                    "uri": f"file://{result_path.absolute()}"
                })
        
        return experiments[offset:offset + limit]

def get_storage():
    """
    Get storage backend with local fallback.
    
    Returns GCS in production, local filesystem in development.
    """
    from src.utils.settings import settings
    
    # Try GCS first
    if settings.GCS_BUCKET:
        try:
            client = storage.Client(project=settings.PROJECT_ID)
            bucket = client.bucket(settings.GCS_BUCKET)
            logger.info(f"Using GCS bucket: {settings.GCS_BUCKET}")
            return GCSStorageBackend(bucket)
        except Exception as e:
            logger.warning(f"GCS initialization failed, falling back to local: {e}")
    
    # Fallback to local storage
    logger.info("Using local storage backend (dev mode)")
    return LocalStorageBackend()
```

**Tests**:
```python
# app/tests/test_storage.py

def test_local_storage_store_and_list(tmp_path):
    """Test local storage backend."""
    storage = LocalStorageBackend(base_path=str(tmp_path))
    
    # Store experiment
    result = {"value": 0.42, "uncertainty": 0.05}
    metadata = {"timestamp": "2025-10-01T00:00:00Z"}
    
    uri = storage.store_experiment_result("test-001", result, metadata)
    
    assert uri.startswith("file://")
    assert (tmp_path / "test-001" / "result.json").exists()
    
    # List experiments
    experiments = storage.list_experiments()
    assert len(experiments) == 1
    assert experiments[0]["experiment_id"] == "test-001"
    assert experiments[0]["result"]["value"] == 0.42

def test_storage_api_with_local_backend(client, monkeypatch):
    """Test storage API endpoints with local backend."""
    # Mock settings to force local storage
    monkeypatch.setenv("GCS_BUCKET", "")
    
    # Store experiment
    response = client.post(
        "/api/storage/experiment",
        json={
            "experiment_id": "test-002",
            "result": {"value": 0.8},
            "metadata": {"method": "BO"}
        }
    )
    
    assert response.status_code == 200
    assert "uri" in response.json()
    
    # List experiments
    response = client.get("/api/storage/experiments")
    assert response.status_code == 200
    data = response.json()
    assert len(data["experiments"]) >= 1
```

**Benefits**:
- ✅ Can test storage endpoints locally
- ✅ No GCP credentials required for dev
- ✅ Same API contract (transparent swap)

---

#### 4. Dual-Model Streaming Hardening 🔧 (3-4 hours)
**Priority**: MEDIUM-HIGH - Improves production reliability  
**Effort**: 3-4 hours  
**Impact**: MEDIUM - Better error handling + audit trail

**Problem**: SSE can hang on partial failures, no experiment persistence

**Implementation**:
```python
# app/src/api/main.py

@app.post("/api/reasoning/query", tags=["reasoning"])
async def query_with_feedback(
    request: QueryRequest,
    background_tasks: BackgroundTasks
):
    """
    Query with dual-model feedback (streaming).
    
    Enhanced with:
    - Proper error handling (no hanging streams)
    - Experiment run persistence
    - Structured error events
    """
    if agent is None:
        return JSONResponse(
            status_code=503,
            content={"error": "AI service unavailable", "code": "vertex_not_initialized"}
        )

    async def event_generator():
        flash_response = None
        pro_response = None
        error_occurred = False
        
        try:
            yield sse_event("start", {
                "query": request.query,
                "flash_model": settings.GEMINI_FLASH_MODEL,
                "pro_model": settings.GEMINI_PRO_MODEL
            })
            
            # Use the public parallel query method
            async for event_type, data in agent.query_parallel_stream(request.query):
                if event_type == "flash_complete":
                    flash_response = data["response"]
                    yield sse_event("flash_complete", data)
                elif event_type == "pro_complete":
                    pro_response = data["response"]
                    yield sse_event("pro_complete", data)
                elif event_type == "error":
                    error_occurred = True
                    yield sse_error(
                        data["error"],
                        code=data.get("code", "unknown_error")
                    )
                    break
            
            if not error_occurred:
                yield sse_event("complete", {
                    "flash_response": flash_response,
                    "pro_response": pro_response
                })
                
                # Persist experiment run in background
                background_tasks.add_task(
                    persist_experiment_run,
                    query=request.query,
                    flash_response=flash_response,
                    pro_response=pro_response
                )
        
        except asyncio.CancelledError:
            logger.warning("Query cancelled by client")
            yield sse_error("Request cancelled", code="cancelled")
            raise
        
        except Exception as e:
            logger.exception("Query failed")
            yield sse_error(
                "Internal server error",
                code="internal_error"
            )

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"  # Disable nginx buffering
        }
    )

async def persist_experiment_run(
    query: str,
    flash_response: str,
    pro_response: str
):
    """Background task to persist experiment run."""
    try:
        db = get_db()
        if db:
            await db.log_experiment_run(
                experiment_id=f"query-{datetime.now().isoformat()}",
                agent_type="dual-gemini",
                prompt=query,
                response=pro_response,  # Use Pro as primary
                metadata={
                    "flash_response": flash_response,
                    "pro_response": pro_response,
                    "flash_model": settings.GEMINI_FLASH_MODEL,
                    "pro_model": settings.GEMINI_PRO_MODEL
                }
            )
            logger.info("Experiment run persisted")
    except Exception as e:
        logger.error(f"Failed to persist experiment run: {e}")
```

```python
# app/src/reasoning/dual_agent.py

async def query_parallel_stream(
    self,
    query: str
) -> AsyncIterator[Tuple[str, Dict[str, Any]]]:
    """
    Query both models in parallel with streaming events.
    
    Yields:
        (event_type, data) tuples:
        - ("flash_complete", {"response": str, "duration_ms": int})
        - ("pro_complete", {"response": str, "duration_ms": int})
        - ("error", {"error": str, "code": str, "model": str})
    """
    try:
        # Create tasks for both models
        flash_task = asyncio.create_task(self._query_flash(query))
        pro_task = asyncio.create_task(self._query_pro(query))
        
        # Wait for flash (should be faster)
        try:
            flash_start = time.time()
            flash_response = await flash_task
            flash_duration = (time.time() - flash_start) * 1000
            
            yield ("flash_complete", {
                "response": flash_response,
                "duration_ms": flash_duration,
                "model": self.flash_model_name
            })
        except Exception as e:
            logger.error(f"Flash model failed: {e}")
            # Cancel Pro task if Flash failed
            pro_task.cancel()
            yield ("error", {
                "error": "Flash model failed",
                "code": "flash_error",
                "model": "flash"
            })
            return
        
        # Wait for Pro
        try:
            pro_start = time.time()
            pro_response = await pro_task
            pro_duration = (time.time() - pro_start) * 1000
            
            yield ("pro_complete", {
                "response": pro_response,
                "duration_ms": pro_duration,
                "model": self.pro_model_name
            })
        except Exception as e:
            logger.error(f"Pro model failed: {e}")
            yield ("error", {
                "error": "Pro model failed",
                "code": "pro_error",
                "model": "pro"
            })
            return
            
    except asyncio.CancelledError:
        # Cleanup: cancel both tasks
        flash_task.cancel()
        pro_task.cancel()
        raise
```

**Tests**:
```python
# app/tests/test_reasoning_stream.py

import pytest
import httpx
from httpx_sse import aconnect_sse

@pytest.mark.asyncio
async def test_sse_stream_success():
    """Test successful SSE stream with both models."""
    async with httpx.AsyncClient() as client:
        async with aconnect_sse(
            client,
            "POST",
            "http://localhost:8080/api/reasoning/query",
            json={"query": "Test query"}
        ) as event_source:
            events = []
            async for sse in event_source.aiter_sse():
                events.append((sse.event, sse.data))
                if sse.event == "complete":
                    break
            
            # Verify event sequence
            assert events[0][0] == "start"
            assert any(e[0] == "flash_complete" for e in events)
            assert any(e[0] == "pro_complete" for e in events)
            assert events[-1][0] == "complete"

@pytest.mark.asyncio
async def test_sse_stream_error_handling():
    """Test SSE stream handles errors gracefully."""
    # Mock agent to raise exception
    async with httpx.AsyncClient() as client:
        async with aconnect_sse(
            client,
            "POST",
            "http://localhost:8080/api/reasoning/query",
            json={"query": "Test query"}
        ) as event_source:
            events = []
            async for sse in event_source.aiter_sse():
                events.append((sse.event, sse.data))
                if sse.event == "error":
                    break
            
            # Verify error event sent
            assert any(e[0] == "error" for e in events)
```

**Benefits**:
- ✅ No hanging streams on partial failures
- ✅ Experiment runs persisted to database
- ✅ Structured error events for clients
- ✅ Better test coverage

---

## 📅 Suggested Timeline

### **Day 1** (Wednesday) - Critical Fixes
- Morning: Task 1 (Health Check) - 1 hour
- Afternoon: Task 2 (Vertex Init) - 2 hours
- **Deploy to staging and verify**

### **Day 2** (Thursday) - Developer Experience  
- Morning: Task 3 (Local Storage) - 3 hours
- Afternoon: Test all storage endpoints locally

### **Day 3** (Friday) - Production Polish
- Morning-Afternoon: Task 4 (SSE Streaming) - 4 hours
- **Deploy to production with full test suite**

**After Weekend**: Start Phase 2A Sprint 1 (Cost Model) fresh on Monday

---

## 🎯 Success Metrics

### Task 1: Health Check
- ✅ Both `/health` and `/healthz` return 200
- ✅ Cloud Run health probes succeed
- ✅ README and deployment docs aligned

### Task 2: Vertex Init
- ✅ Health check exposes Vertex status
- ✅ Retry endpoint works
- ✅ No silent failures

### Task 3: Local Storage
- ✅ Storage endpoints work without GCS
- ✅ Can test full workflow locally
- ✅ 100% storage test coverage

### Task 4: SSE Streaming
- ✅ No hanging streams on errors
- ✅ Experiment runs persist to DB
- ✅ Test coverage > 80% on reasoning endpoint

---

## 🔄 Integration with Phase 2 Roadmap

**These tasks are prerequisites for Phase 2A** because:

1. **Health Check** → Needed for production reliability during Phase 2 deployments
2. **Vertex Init** → Critical for ops during intensive RL training
3. **Local Storage** → Enables fast iteration on cost model without GCP
4. **SSE Streaming** → Foundation for real-time RL monitoring dashboard (Week 2)

**Recommendation**: Complete these 4 tasks (2-3 days) before starting Phase 2A Sprint 1

---

## 📝 Commit Strategy

### Commit 1: Health Check + Vertex Init
```bash
git checkout -b feat/production-hardening
# Implement tasks 1 & 2
git add app/src/api/main.py app/src/services/vertex.py app/tests/
git commit -m "feat: Add /healthz alias and expose Vertex init status

- Add /healthz endpoint alias for Cloud Run compatibility
- Surface Vertex initialization errors in health check
- Add admin retry endpoint for Vertex recovery
- Prevent silent failures with structured error tracking

Fixes: #1 (health check mismatch)
Fixes: #4 (vertex init errors)
"
```

### Commit 2: Local Storage
```bash
# Implement task 3
git add app/src/services/storage.py app/tests/test_storage.py
git commit -m "feat: Add local filesystem storage backend

- Implement LocalStorageBackend for dev environments
- Auto-fallback when GCS credentials unavailable
- Enable local testing of storage endpoints
- Add comprehensive storage tests

Fixes: #3 (local storage)
"
```

### Commit 3: SSE Streaming
```bash
# Implement task 4
git add app/src/api/main.py app/src/reasoning/dual_agent.py app/tests/
git commit -m "feat: Harden SSE streaming and persist experiments

- Add query_parallel_stream() for structured error handling
- Prevent hanging streams on partial failures
- Persist experiment runs to database
- Add async streaming tests with httpx-sse

Fixes: #2 (streaming hardening)
"
```

### Deploy
```bash
git push origin feat/production-hardening
# Create PR, review, merge
# Deploy to staging, then production
```

---

## 🚀 Quick Start Commands

### Setup
```bash
cd /Users/kiteboard/periodicdent42
git checkout -b feat/production-hardening
git pull origin main
```

### Task 1: Health Check (Start Here!)
```bash
# Edit health endpoint
code app/src/api/main.py

# Add tests
code app/tests/test_health.py

# Run tests
cd app && pytest tests/test_health.py -v

# Test locally
curl http://localhost:8080/health
curl http://localhost:8080/healthz
```

### Task 2: Vertex Init
```bash
# Add status tracking
code app/src/services/vertex.py

# Update health check
code app/src/api/main.py

# Test
cd app && pytest tests/test_vertex_init.py -v
```

### Task 3: Local Storage
```bash
# Implement local backend
code app/src/services/storage.py

# Add tests
code app/tests/test_storage.py

# Test without GCS
unset GCS_BUCKET
cd app && pytest tests/test_storage.py -v
```

### Task 4: SSE Streaming
```bash
# Add streaming helper
code app/src/reasoning/dual_agent.py

# Update endpoint
code app/src/api/main.py

# Add tests
pip install httpx-sse
code app/tests/test_reasoning_stream.py
cd app && pytest tests/test_reasoning_stream.py -v
```

---

## ❓ Questions to Consider

1. **Health Check**: Should `/healthz` be the primary endpoint and `/health` the alias? (Cloud Run convention is `/healthz`)

2. **Vertex Retry**: Should retry endpoint require additional auth beyond API key? (e.g., admin-only token)

3. **Local Storage**: Where should local files be stored? `./local_storage` or `/tmp`? Should they persist across restarts?

4. **SSE Persistence**: Should failed experiments also be persisted? How to handle partial data?

---

## 🎉 Benefits Summary

**After completing these 4 tasks**:
- ✅ Production-ready health checks (no false alarms)
- ✅ Observable Vertex failures (faster recovery)
- ✅ Local development without GCP (faster iteration)
- ✅ Reliable streaming with audit trail (production confidence)

**Ready for**: Phase 2A Sprint 1 (Cost Model) with solid foundation

---

**Recommendation**: Start with Task 1 (health check) now - it's a 1-hour fix with immediate production impact! 🚀


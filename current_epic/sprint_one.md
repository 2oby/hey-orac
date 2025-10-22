# Sprint One: System Hardening & Lifecycle Management

**Epic**: Production Readiness & Security Hardening
**Sprint Duration**: TBD
**Priority**: HIGH
**Status**: Planning

---

## Task 1: Robust Lifecycle Management & Thread Monitoring

### Priority: HIGH
**Estimated Effort**: 8-12 hours

### Problem Statement

**Root Cause Identified**: 2025-10-22 - Audio thread became unresponsive after 1115 failed restart attempts when PyAudio's `stream.read()` call blocked indefinitely at `audio_reader_thread.py:198`. The blocking call prevented the restart mechanism from working because:

1. The `AudioReaderThread.restart()` method couldn't interrupt the blocking `stream.read()` call
2. Thread appeared "alive" but was stuck waiting for audio data that never came
3. RMS values were stuck at 79 for over an hour (~4030 seconds)
4. Likely caused by transient USB audio device issue or hardware glitch

**Current Limitations**:
- Existing watchdog mechanisms (periodic health checks every 5s, stuck RMS detection) couldn't handle completely blocking system calls
- Thread join timeout of 2.0 seconds wasn't sufficient to detect the blocking issue
- No external process supervision to force-restart the container

### Requirements

#### 1.1 Audio Thread Watchdog Enhancement

**Objective**: Detect and recover from blocking `stream.read()` calls

**Implementation**:
- Add dedicated watchdog thread that monitors `last_read_time` independently
- If `last_read_time` exceeds threshold (e.g., 10 seconds), forcefully terminate and recreate the audio reader thread
- Use `threading.Thread.is_alive()` combined with timeout detection
- Log detailed diagnostics before forced termination (thread stack traces if possible)

**Files to Modify**:
- `src/hey_orac/audio/audio_reader_thread.py`
- `src/hey_orac/wake_word_detection.py` (watchdog initialization)

**Success Criteria**:
- Audio thread automatically recovers from blocking calls within 15 seconds
- No manual intervention required for USB device glitches
- Detailed logging of recovery events for debugging

#### 1.2 Docker Health Check Enhancement

**Objective**: External container-level monitoring with automatic restart

**Current Health Check** (docker-compose.yml line 43-48):
```yaml
healthcheck:
  test: ["CMD", "python3", "-c", "import openwakeword; print('OK')"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 40s
```

**Problems**:
- Only checks if OpenWakeWord can be imported, not if audio is actually flowing
- Doesn't detect stuck RMS values or frozen audio processing

**New Health Check Implementation**:
- Create dedicated health check script (`scripts/healthcheck.py`)
- Check multiple indicators:
  - RMS values changing (not stuck for > 30 seconds)
  - Audio thread last read time (< 15 seconds old)
  - Web server responding to `/api/health` endpoint
  - Recent chunk processing (checked via shared state or temp file)
- Return exit code 0 (healthy) or 1 (unhealthy)
- Docker will automatically restart container after 3 consecutive failures

**Files to Create**:
- `scripts/healthcheck.py`

**Files to Modify**:
- `docker-compose.yml` (update healthcheck command)
- `src/hey_orac/wake_word_detection.py` (expose health metrics)

**Success Criteria**:
- Container automatically restarts when audio processing is stuck
- Health check runs every 30 seconds
- No false positives during normal operation

#### 1.3 Memory Leak Detection & Monitoring

**Objective**: Detect and prevent memory leaks from long recordings or resource buildup

**Current Status**:
- Pi has 14GB free memory (plenty available)
- Recording directory nearly empty (4KB)
- No evidence of current memory leaks
- Max recording duration: 15 seconds (~960KB per recording)

**Implementation**:
- Add periodic memory usage monitoring (every 5 minutes)
- Track memory growth trends over time
- Log warnings if memory usage exceeds thresholds:
  - WARNING: > 400 MB (expected: ~150-200 MB)
  - CRITICAL: > 450 MB (trigger graceful restart)
- Proper cleanup verification after each recording:
  - Unregister speech recorder consumer
  - Clear audio chunks from memory
  - Verify numpy arrays are freed

**Files to Modify**:
- `src/hey_orac/audio/speech_recorder.py` (verify cleanup)
- `src/hey_orac/wake_word_detection.py` (add memory monitoring)

**Success Criteria**:
- Memory usage stays below 250 MB during normal operation
- No memory growth over 24+ hour runs
- Automatic graceful restart if memory leak detected

#### 1.4 Process-Level Supervision

**Objective**: Add external process supervisor for critical failure recovery

**Options to Evaluate**:
1. **supervisord** - Industry standard process supervisor
2. **systemd** (if running outside Docker)
3. **Docker restart policies** (already in place: `restart: unless-stopped`)

**Current Docker Restart Policy**: Already configured (docker-compose.yml line 8)

**Enhancement**:
- Ensure restart policy is robust
- Add logging for restart events
- Track restart count and reason
- Alert on excessive restarts (> 5 per hour indicates deeper issue)

**Files to Modify**:
- `docker-compose.yml` (verify restart policy)
- Documentation for restart monitoring

**Success Criteria**:
- Container automatically restarts on crashes
- Restart events logged and trackable
- System recovers from any failure within 60 seconds

#### 1.5 Audio Device Recovery

**Objective**: Better handling of USB device disconnection/reconnection

**Implementation**:
- Detect PyAudio stream failures (OSError, IOError)
- Attempt to re-enumerate audio devices
- Recreate PyAudio stream with fresh device connection
- Maximum retry attempts: 3
- Delay between retries: 2 seconds

**Files to Modify**:
- `src/hey_orac/audio/audio_manager.py`
- `src/hey_orac/audio/audio_reader_thread.py`

**Success Criteria**:
- System recovers from USB microphone unplug/replug within 10 seconds
- Audio device changes detected and handled gracefully
- Detailed logging of device recovery events

### Testing Requirements

1. **Stress Testing**:
   - Run for 24+ hours continuously
   - Simulate USB device disconnection
   - Simulate blocking system calls (mock `stream.read()`)

2. **Chaos Engineering**:
   - Kill audio thread randomly
   - Disconnect network during STT calls
   - Fill disk to trigger write failures

3. **Health Check Validation**:
   - Verify health check detects stuck RMS within 90 seconds
   - Confirm container restarts automatically
   - Check no false positives during normal operation

### Documentation Updates

- Update DEVELOPER_GUIDE.md with new health monitoring section
- Document recovery mechanisms in USER_GUIDE.md
- Add troubleshooting section for stuck audio issues
- Update devlog.md with sprint completion notes

---

## Task 2: Security Hardening - Authentication & Authorization

### Priority: HIGH
**Estimated Effort**: 16-24 hours

### Problem Statement

**Current Security Posture**:
- No authentication on web GUI (port 7171)
- No authentication on API endpoints
- No authentication on heartbeat calls
- No encryption on internal API calls
- System exposed on local network without access controls

**Security Risks**:
- Unauthorized access to wake word detection system
- Ability to modify configuration without authentication
- Potential eavesdropping on voice commands
- No audit trail of who made changes
- Vulnerable to local network attacks

### Security Model Design

#### 2.1 Define ORAC Stack Security Model

**Scope**: Security across all ORAC components:
1. **Hey ORAC** (this module) - Wake word detection
2. **ORAC STT** - Speech-to-text service
3. **ORAC Core** - Main orchestration (future component)

**Security Architecture**:
```
┌─────────────────────────────────────────────────────┐
│              ORAC Stack Security Model               │
├─────────────────────────────────────────────────────┤
│                                                       │
│  ┌──────────────┐         ┌──────────────┐          │
│  │  Hey ORAC    │  Auth   │  ORAC STT    │          │
│  │  (Web GUI)   │◄───────►│  (API)       │          │
│  │  Port 7171   │  Token  │  Port 7272   │          │
│  └──────┬───────┘         └──────┬───────┘          │
│         │                         │                   │
│         │  Auth Token             │  Auth Token       │
│         │                         │                   │
│  ┌──────▼─────────────────────────▼───────┐          │
│  │         ORAC Core (Future)             │          │
│  │    Central Auth & Orchestration        │          │
│  │         Port: TBD                      │          │
│  └────────────────────────────────────────┘          │
│                                                       │
└─────────────────────────────────────────────────────┘
```

**Authentication Strategy**:
- **Option A**: Shared secret token (simple, good for single-user)
- **Option B**: JWT tokens with expiration (scalable, good for multi-user)
- **Option C**: OAuth2 / OpenID Connect (enterprise-grade, complex)

**Recommendation**: Start with **Option B (JWT)** for balance of security and complexity

#### 2.2 Web GUI Authentication

**Objective**: Add login page to Hey ORAC web interface

**Implementation**:
- Login page with username/password
- Session management with secure cookies
- JWT token issuance on successful login
- Token expiration and refresh mechanism
- Logout functionality

**User Management**:
- Initial simple approach: Single admin user
- Credentials stored in config file (hashed with bcrypt)
- Future: Multi-user with roles (admin, viewer)

**Files to Create**:
- `src/hey_orac/web/auth.py` - Authentication logic
- `src/hey_orac/web/static/login.html` - Login page
- `src/hey_orac/web/static/js/auth.js` - Client-side auth handling

**Files to Modify**:
- `src/hey_orac/web/app.py` - Add auth middleware
- `src/hey_orac/web/routes.py` - Protect endpoints with @require_auth decorator
- `src/hey_orac/config/manager.py` - Add auth config section
- `config/settings.json.template` - Add auth settings

**Configuration Schema Addition**:
```json
{
  "auth": {
    "enabled": true,
    "admin_username": "admin",
    "admin_password_hash": "$2b$12$...",
    "jwt_secret": "generate-random-secret",
    "jwt_expiration_hours": 24,
    "require_https": false
  }
}
```

**Protected Endpoints**:
- `PUT /api/config/global` - Require authentication
- `PUT /api/config/models/{name}` - Require authentication
- `GET /api/models` - Allow unauthenticated (read-only)
- `GET /api/health` - Allow unauthenticated (monitoring)

**Success Criteria**:
- Login page displays on accessing web GUI
- Valid credentials grant access
- Invalid credentials rejected with clear error
- Session persists across page refreshes
- Auto-logout after token expiration

#### 2.3 API Authentication

**Objective**: Secure all API endpoints with token-based auth

**Implementation**:
- All API calls require `Authorization: Bearer <token>` header
- Token validation on each request
- Rate limiting to prevent brute force (future enhancement)
- Audit logging of all authenticated requests

**Token Format**:
```
JWT Token Structure:
{
  "sub": "admin",
  "iat": 1234567890,
  "exp": 1234654290,
  "role": "admin"
}
```

**Files to Modify**:
- `src/hey_orac/web/routes.py` - Add auth decorators to all endpoints
- `src/hey_orac/web/app.py` - Add token validation middleware

**Success Criteria**:
- Unauthenticated API calls return 401 Unauthorized
- Valid tokens grant access
- Expired tokens rejected
- Malformed tokens rejected

#### 2.4 Heartbeat Authentication

**Objective**: Secure heartbeat calls between Hey ORAC and ORAC STT

**Current Heartbeat** (`src/hey_orac/transport/heartbeat_sender.py`):
- Sends periodic heartbeat to ORAC STT at `/stt/v1/heartbeat`
- No authentication currently

**Implementation**:
- Add `Authorization` header to heartbeat requests
- Use shared secret or JWT token
- ORAC STT validates token before accepting heartbeat
- Log authentication failures

**Files to Modify**:
- `src/hey_orac/transport/heartbeat_sender.py`
- `src/hey_orac/transport/stt_client.py` (for STT API calls)

**Configuration Addition**:
```json
{
  "stt": {
    "url": "http://192.168.8.192:7272/stt/v1/stream",
    "auth_token": "shared-secret-or-jwt",
    "auth_enabled": true
  }
}
```

**Success Criteria**:
- Heartbeats include authentication token
- ORAC STT accepts authenticated heartbeats
- Unauthenticated heartbeats rejected by ORAC STT
- Authentication failures logged

#### 2.5 Inter-Service Communication Security

**Objective**: Define security model for all ORAC component communication

**Components**:
1. **Hey ORAC → ORAC STT**: Audio streaming, heartbeats
2. **ORAC STT → ORAC Core**: Transcription results, status
3. **ORAC Core → Hey ORAC**: Configuration updates, commands

**Security Requirements**:
- **Confidentiality**: Consider TLS for sensitive data (voice recordings)
- **Integrity**: Verify message authenticity (HMAC or JWT signatures)
- **Authentication**: Mutual authentication between services
- **Authorization**: Role-based access control

**Implementation Options**:

**Option 1: Shared Secret** (Simple, internal network)
- Each service has pre-shared secret
- HMAC-based request signing
- Good for: Single-user, local network deployment

**Option 2: JWT with Service Accounts** (Recommended)
- Each service has service account credentials
- JWT tokens with short expiration
- Token refresh mechanism
- Good for: Multi-component deployments

**Option 3: mTLS (Mutual TLS)** (Enterprise)
- Client certificates for each service
- Automatic encryption and authentication
- Certificate management overhead
- Good for: Production, multi-tenant environments

**Recommendation**: Start with **Option 2 (JWT Service Accounts)**

**Service Account Configuration**:
```json
{
  "service_accounts": {
    "hey_orac": {
      "id": "hey_orac_service",
      "secret": "generate-random-secret",
      "permissions": ["stt.transcribe", "stt.heartbeat"]
    },
    "orac_stt": {
      "id": "orac_stt_service",
      "secret": "generate-random-secret",
      "permissions": ["core.results", "core.status"]
    }
  }
}
```

**Files to Create**:
- `src/hey_orac/auth/service_account.py` - Service account management
- `src/hey_orac/auth/token_manager.py` - JWT token generation/validation

**Success Criteria**:
- All inter-service calls authenticated
- Unauthorized calls rejected
- Token expiration handled gracefully
- Audit trail of all service-to-service calls

#### 2.6 Security Configuration Management

**Objective**: Secure storage and management of credentials

**Requirements**:
- Passwords hashed with bcrypt (cost factor: 12)
- JWT secrets randomly generated (256-bit minimum)
- Service account secrets in secure storage
- No plaintext secrets in git repository
- Environment variables for sensitive data

**Implementation**:
- Use environment variables for secrets in Docker
- Config file contains only hashes, not plaintext
- Secret generation utility script
- Documentation for secure deployment

**Files to Create**:
- `scripts/generate_secrets.py` - Generate secure random secrets
- `.env.template` - Template for environment variables

**Files to Modify**:
- `docker-compose.yml` - Add environment variable injection
- `config/settings.json.template` - Remove plaintext secrets

**Configuration Example**:
```yaml
# docker-compose.yml
environment:
  - JWT_SECRET=${JWT_SECRET}
  - ADMIN_PASSWORD_HASH=${ADMIN_PASSWORD_HASH}
  - STT_AUTH_TOKEN=${STT_AUTH_TOKEN}
```

**Success Criteria**:
- No plaintext passwords in config files
- Secrets can be rotated without code changes
- Environment variables properly injected
- Documentation for secret management

### ORAC Stack Security Coordination

**Cross-Component Changes Required**:

#### ORAC STT Modifications:
1. Add token validation to all endpoints
2. Accept `Authorization` header on heartbeat endpoint
3. Validate service account tokens
4. Return 401 for unauthenticated requests
5. Add audit logging

#### ORAC Core (Future):
1. Central authentication service
2. Token issuance and validation
3. User management
4. Role-based access control
5. Audit log aggregation

#### Hey ORAC (This Module):
1. Implement web GUI login
2. Add API authentication
3. Secure heartbeat calls
4. Service account integration
5. Audit logging

### Testing Requirements

#### Security Testing:
1. **Authentication Tests**:
   - Valid credentials accepted
   - Invalid credentials rejected
   - Expired tokens rejected
   - Malformed tokens rejected

2. **Authorization Tests**:
   - Authenticated users can modify config
   - Unauthenticated users blocked
   - Read-only endpoints accessible

3. **Token Management Tests**:
   - Token generation works
   - Token expiration enforced
   - Token refresh works
   - Logout invalidates tokens

4. **Penetration Testing**:
   - SQL injection attempts (N/A - no SQL)
   - XSS attempts on web GUI
   - CSRF protection
   - Brute force protection

5. **Integration Tests**:
   - Hey ORAC → ORAC STT authenticated calls
   - Service account token exchange
   - End-to-end authenticated flow

### Documentation Updates

- Create SECURITY.md with security model documentation
- Update DEVELOPER_GUIDE.md with authentication section
- Update USER_GUIDE.md with login instructions
- Document credential rotation procedures
- Add security best practices guide

### Dependencies

**Python Packages to Add**:
- `PyJWT==2.8.0` - JWT token handling
- `bcrypt==4.1.0` - Password hashing
- `python-dotenv==1.0.0` - Environment variable management

**Files to Modify**:
- `requirements.txt` - Add security dependencies

---

## Sprint Success Criteria

### Task 1 (Lifecycle Management):
- [ ] Audio thread recovers from blocking calls automatically
- [ ] Docker health check detects stuck processes
- [ ] Memory leaks prevented or detected early
- [ ] System runs stable for 72+ hours
- [ ] Container auto-restarts on failures
- [ ] USB device recovery working

### Task 2 (Security):
- [ ] Web GUI requires login
- [ ] API endpoints require authentication
- [ ] Heartbeat calls authenticated
- [ ] Service account model documented
- [ ] No plaintext secrets in repository
- [ ] Security testing passed
- [ ] ORAC stack security model defined

---

## Sprint Planning Notes

**Dependencies**:
- Task 1 can be started immediately
- Task 2 requires coordination with ORAC STT team
- Both tasks are independent and can run in parallel

**Risk Assessment**:
- Task 1: LOW risk - improvements to existing mechanisms
- Task 2: MEDIUM risk - requires cross-component changes

**Next Sprint Candidates**:
- TLS/HTTPS implementation
- Rate limiting and DDoS protection
- Advanced monitoring and alerting
- Multi-user support with roles
- Centralized logging and audit trails

---

**Created**: 2025-10-22
**Last Updated**: 2025-10-22
**Status**: Ready for Sprint Planning

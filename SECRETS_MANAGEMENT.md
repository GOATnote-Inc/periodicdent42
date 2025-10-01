# Secrets Management Guide

**⚠️ CRITICAL: NEVER COMMIT SECRETS TO GIT**

This guide explains how to properly handle secrets in development and production.

---

## 🚨 CRITICAL SECURITY WARNINGS

### Terminal History Leak Prevention

**NEVER** run commands that echo secrets in plain text:
```bash
# ❌ BAD - Leaves secret in history
echo "API_KEY=abc123..." >> .env
export API_KEY="abc123..."

# ✅ GOOD - Use our secure scripts
bash scripts/init_secrets_and_env.sh
bash scripts/rotate_api_key.sh
```

**Why?** Your terminal history (`.bash_history`, `.zsh_history`) can be accessed by:
- Malicious actors with local access
- Screen recording/sharing tools  
- CI/CD logs if run in pipelines
- Shell history commands (`history`, `fc -l`)

**Protection:**
1. ✅ Our scripts only show **masked keys**: `abc12345...xyz78901`
2. ✅ Full keys saved to **chmod 600 files** only
3. ✅ Use `cat .api-key` when you need the full key

### Clear Sensitive History

If you accidentally typed a secret in terminal:
```bash
# Clear current session history
history -c

# Clear history file
echo "" > ~/.zsh_history  # or ~/.bash_history
source ~/.zshrc

# On macOS, also clear:
rm ~/.bash_sessions/*  # if using bash
```

---

## 🔒 Security Principles

1. **NEVER hardcode secrets** in source code
2. **NEVER commit secrets** to git (even in private repos)
3. **Use Secret Manager** for production secrets
4. **Use environment variables** for local development
5. **Rotate secrets regularly** (API keys every 90 days)
6. **Use different secrets** for dev/staging/prod environments

---

## 📁 Files You Need

### For Local Development

```bash
app/.env                    # Your local secrets (NOT in git)
app/env.example             # Template to copy (IN git)
app/env.production.example  # Production reference (IN git)
```

### Helper Scripts

```bash
scripts/init_secrets_and_env.sh  # 🚀 Initialize: retrieve OR generate secrets (safe)
scripts/rotate_api_key.sh        # 🔄 Force generate NEW API key (rotation)
scripts/setup_local_dev.sh       # Retrieve existing secrets only
scripts/get_secrets.sh           # View secrets from Secret Manager
```

**Important**: `init_secrets_and_env.sh` retrieves existing secrets (doesn't rotate).  
**To force a new key**: Use `rotate_api_key.sh`

---

## 🛠️ Local Development Setup

### Option 1: Full Auto-Init (Recommended) 🚀

```bash
# This does EVERYTHING - generates NEW secrets if needed!
bash scripts/init_secrets_and_env.sh
```

This will:
1. ✅ Check if secrets exist in Secret Manager
2. ✅ **Generate NEW secrets if they don't exist** (using `openssl rand`)
3. ✅ Create secrets in Secret Manager automatically
4. ✅ Create `app/.env` with all values populated
5. ✅ Save API key to `.api-key` for reference
6. ✅ Set proper permissions (chmod 600)

**Perfect for**: First-time setup or fresh environments

### Option 2: Retrieve Existing Secrets

```bash
# Use this if secrets already exist in Secret Manager
bash scripts/setup_local_dev.sh
```

This will:
1. ✅ Copy `env.example` to `.env`
2. ✅ Retrieve existing secrets from Secret Manager
3. ✅ Update `.env` with the values
4. ✅ Give you instructions to start the server

**Perfect for**: Team members joining an existing project

### Option 3: Manual Setup

```bash
# 1. Copy the example file
cp app/env.example app/.env

# 2. Get your API key from Secret Manager
export API_KEY=$(gcloud secrets versions access latest --secret=api-key --project=periodicdent42)

# 3. Add it to your .env file
echo "API_KEY=$API_KEY" >> app/.env

# 4. Or just export it in your terminal
export API_KEY="your-key-here"
```

### Option 3: Environment Variables Only

```bash
# Set in your terminal (not persistent)
export PROJECT_ID=periodicdent42
export LOCATION=us-central1
export ENVIRONMENT=development
export ENABLE_AUTH=false
export API_KEY=$(gcloud secrets versions access latest --secret=api-key --project=periodicdent42)

# Start the server
cd app
uvicorn src.api.main:app --reload --port 8080
```

---

## 🔐 Retrieving Secrets from Secret Manager

### View Available Secrets

```bash
bash scripts/get_secrets.sh
```

### Get Specific Secret

```bash
# API Key
gcloud secrets versions access latest --secret=api-key --project=periodicdent42

# Database Password
gcloud secrets versions access latest --secret=DB_PASSWORD --project=periodicdent42

# GCS Bucket
gcloud secrets versions access latest --secret=GCS_BUCKET --project=periodicdent42

# SQL Instance
gcloud secrets versions access latest --secret=GCP_SQL_INSTANCE --project=periodicdent42
```

### List All Secrets

```bash
gcloud secrets list --project=periodicdent42
```

---

## 🔄 Rotating API Keys

### When to Rotate

- **Regular schedule**: Every 90 days
- **Security incident**: Immediately
- **Suspected compromise**: Immediately  
- **Team member departure**: Within 24 hours

### Automated Rotation (Recommended)

```bash
# Run the rotation script
bash scripts/rotate_api_key.sh
```

This will:
1. Generate a NEW cryptographically secure API key
2. Add it as a new version in Secret Manager
3. Update your local `.api-key` and `app/.env` files
4. Show you the next steps

The script prompts you before making changes and keeps the old key active until you disable it.

### After Rotation

```bash
# 1. Restart Cloud Run to use new key
gcloud run services update ard-backend --region=us-central1 --project=periodicdent42

# 2. Distribute new key to all authorized clients

# 3. After 24 hours (when all clients updated), disable old version
gcloud secrets versions list api-key --project=periodicdent42
gcloud secrets versions disable VERSION_NUMBER --secret=api-key --project=periodicdent42
```

**Set calendar reminder**: December 30, 2025 (90 days from October 1, 2025)

---

## ☁️ Production (Cloud Run)

In production, **DO NOT create .env files**. Secrets are automatically loaded from Secret Manager.

### How It Works

The deployment script (`infra/scripts/deploy_cloudrun.sh`) uses:

```bash
gcloud run deploy ard-backend \
  --set-env-vars="PROJECT_ID=periodicdent42,ENABLE_AUTH=true" \
  --set-secrets="API_KEY=api-key:latest"
```

This tells Cloud Run to:
1. ✅ Load `API_KEY` from Secret Manager secret `api-key` (latest version)
2. ✅ Make it available as environment variable `API_KEY`
3. ✅ Keep it secure (never logged or exposed)

### Updating Production Secrets

```bash
# 1. Generate new secret
NEW_KEY=$(openssl rand -hex 32)

# 2. Add new version to Secret Manager
echo -n "$NEW_KEY" | gcloud secrets versions add api-key --data-file=-

# 3. Restart Cloud Run (automatically uses latest version)
gcloud run services update ard-backend --region=us-central1 --project=periodicdent42

# 4. Disable old version after verifying
gcloud secrets versions list api-key
gcloud secrets versions disable VERSION_NUMBER --secret=api-key
```

---

## 📋 Environment Variables Reference

### Required for Production

| Variable | Description | Source |
|----------|-------------|--------|
| `PROJECT_ID` | GCP project ID | Set directly |
| `LOCATION` | GCP region | Set directly |
| `ENVIRONMENT` | Environment name | Set directly |
| `ENABLE_AUTH` | Enable authentication | Set directly |
| `API_KEY` | API authentication key | **Secret Manager** |

### Optional

| Variable | Description | Source |
|----------|-------------|--------|
| `ALLOWED_ORIGINS` | CORS allowed origins | Set directly |
| `RATE_LIMIT_PER_MINUTE` | Rate limit | Set directly |
| `DB_PASSWORD` | Database password | **Secret Manager** |
| `GCP_SQL_INSTANCE` | Cloud SQL instance | **Secret Manager** |
| `GCS_BUCKET` | Storage bucket name | **Secret Manager** |

---

## 🚫 What NOT to Do

### ❌ DON'T: Hardcode Secrets

```python
# BAD - NEVER DO THIS!
API_KEY = "your-secret-key-here-never-hardcode-this"
```

### ❌ DON'T: Commit .env Files

```bash
# BAD - NEVER DO THIS!
git add app/.env
git commit -m "Added config"
```

### ❌ DON'T: Share Secrets via Email/Slack

```
# BAD - NEVER DO THIS!
"Hey, here's the API key: abc123def456..."
```

### ❌ DON'T: Use Production Secrets in Development

```bash
# BAD - Use different keys per environment!
# Don't use the same API key for dev, staging, and prod
```

---

## ✅ What TO Do

### ✅ DO: Use Secret Manager

```bash
# GOOD!
gcloud secrets versions access latest --secret=api-key
```

### ✅ DO: Use Environment Variables

```python
# GOOD!
import os
api_key = os.getenv("API_KEY")
```

### ✅ DO: Use .env Files (Local Only)

```bash
# GOOD - but never commit .env!
# .env is in .gitignore
echo "API_KEY=..." > app/.env
```

### ✅ DO: Rotate Secrets Regularly

```bash
# GOOD - Rotate every 90 days
bash scripts/rotate_api_key.sh  # (if we create this)
```

### ✅ DO: Use Different Keys per Environment

```
Development:   api-key-dev
Staging:       api-key-staging  
Production:    api-key-prod
```

---

## 🔍 Verifying Secrets Are Not Committed

### Check Your Git Status

```bash
git status
# Should NOT show .env or .api-key
```

### Search for Hardcoded Secrets

```bash
# Check if any secrets were accidentally committed
git log -p | grep -i "api[_-]key" | grep -v "API_KEY="
```

### Scan Repository

```bash
# Use tools like gitleaks or truffleHog
pip install detect-secrets
detect-secrets scan
```

---

## 🔄 Secret Rotation Schedule

| Secret | Rotation Frequency | Last Rotated | Next Rotation |
|--------|-------------------|--------------|---------------|
| API Key | Every 90 days | 2025-10-01 | 2025-12-30 |
| DB Password | Every 180 days | TBD | TBD |
| GCS Keys | Every 180 days | TBD | TBD |

**Set calendar reminders for rotation dates!**

---

## 📞 If a Secret is Compromised

### Immediate Actions

1. **Rotate immediately**:
   ```bash
   NEW_KEY=$(openssl rand -hex 32)
   echo -n "$NEW_KEY" | gcloud secrets versions add api-key --data-file=-
   ```

2. **Force restart**:
   ```bash
   gcloud run services update ard-backend --region=us-central1
   ```

3. **Disable old version**:
   ```bash
   gcloud secrets versions disable OLD_VERSION --secret=api-key
   ```

4. **Check logs for unauthorized access**:
   ```bash
   gcloud logging read 'resource.type="cloud_run_revision" AND httpRequest.status=401' --limit=100
   ```

5. **Notify team and update all clients**

---

## 📚 Additional Resources

- [Google Secret Manager Docs](https://cloud.google.com/secret-manager/docs)
- [12-Factor App Config](https://12factor.net/config)
- [OWASP Secrets Management](https://cheatsheetseries.owasp.org/cheatsheets/Secrets_Management_Cheat_Sheet.html)

---

## 🆘 Common Issues

### "Permission denied" when accessing secrets

```bash
# Solution: Authenticate with application default credentials
gcloud auth application-default login
```

### "Secret not found"

```bash
# Solution: Check if secret exists
gcloud secrets list --project=periodicdent42

# Create if missing
bash infra/scripts/create_secrets.sh
```

### .env file not being read

```bash
# Solution: Use python-dotenv or export manually
pip install python-dotenv

# Or export manually
export $(cat app/.env | xargs)
```

---

**Remember**: When in doubt, use Secret Manager. Never hardcode secrets!

**Quick Start**:
```bash
bash scripts/setup_local_dev.sh
```


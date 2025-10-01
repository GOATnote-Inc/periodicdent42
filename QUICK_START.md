# 🚀 Quick Start Guide

**Get up and running in 2 minutes!**

---

## ⚡ Fastest Way (One Command)

```bash
# This does EVERYTHING automatically:
# • Retrieves existing secrets from Secret Manager
# • OR generates NEW secrets if they don't exist yet
# • Creates your .env file
bash scripts/init_secrets_and_env.sh
```

**Note**: If secrets already exist, it retrieves them (doesn't rotate).  
**To force a NEW API key**, use: `bash scripts/rotate_api_key.sh`

**Then start the server:**
```bash
cd app
source venv/bin/activate
uvicorn src.api.main:app --reload --port 8080
```

**Done!** Visit http://localhost:8080/docs

---

## 📋 What the Script Does

The `init_secrets_and_env.sh` script automatically:

1. **Checks Google Secret Manager** for existing secrets
2. **Generates NEW secrets if they don't exist**:
   - API Key: 32-byte random hex (for authentication)
   - DB Password: 32-byte random string
   - SQL Instance: Your project's default
   - GCS Bucket: Your project's default
3. **Creates secrets in Secret Manager**
4. **Populates `app/.env`** with all values
5. **Saves API key** to `.api-key` for reference
6. **Sets permissions** (chmod 600 for security)

**Result**: Your local dev environment is 100% ready!

---

## 🔐 What Gets Generated

After running the script, you'll have:

```
app/.env              # Your environment config (gitignored)
.api-key             # Your API key for reference (gitignored)
```

And in Google Secret Manager:
- `api-key` - Your unique API authentication key
- `DB_PASSWORD` - Database password
- `GCP_SQL_INSTANCE` - Cloud SQL connection string
- `GCS_BUCKET` - Storage bucket name

---

## 📊 Example Output

```bash
$ bash scripts/init_secrets_and_env.sh

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🔐 Initialize Secrets & Environment
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Project: periodicdent42

🔍 Checking/Creating secrets in Secret Manager...

1. API Key (api-key)
  🔄 api-key doesn't exist - creating...
  📝 Value: a7c3f891b2d4e5f6...8a9b0c1d

2. Database Password (DB_PASSWORD)
  🔄 DB_PASSWORD doesn't exist - creating...
  📝 Value: xK9mP2... (hidden)

3. Cloud SQL Instance (GCP_SQL_INSTANCE)
  ✅ GCP_SQL_INSTANCE exists - retrieving...
  📝 Value: periodicdent42:us-central1:ard-db

4. Cloud Storage Bucket (GCS_BUCKET)
  ✅ GCS_BUCKET exists - retrieving...
  📝 Value: periodicdent42-experiments

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
📝 Creating .env file
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

✅ Created app/.env with secrets from Secret Manager
✅ Saved API key to .api-key (chmod 600)

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
🎉 Setup Complete!
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Your environment is ready:

  📁 app/.env        - Environment variables
  🔑 .api-key        - Your API key (for reference)

🚀 Start Development Server:
  cd app
  source venv/bin/activate
  uvicorn src.api.main:app --reload --port 8080

Then visit: http://localhost:8080/docs
```

---

## 🔄 Understanding Secret Behavior

### The Init Script (`init_secrets_and_env.sh`)

**Smart retrieval** - doesn't rotate existing keys:
- ✅ If secrets exist → retrieves them
- ✅ If secrets are missing → generates new ones
- ✅ Never overwrites existing secrets
- ✅ Safe to run multiple times

**Use when**: First time setup or joining existing project

### Rotating Keys (`rotate_api_key.sh`)

**Forces new key generation**:
- 🔄 Always generates a NEW API key
- 📝 Adds new version to Secret Manager
- 🔒 Keeps old version (can disable later)
- ⚠️ Requires updating all clients

**Use when**: 
- Key rotation schedule (every 90 days)
- Security incident
- Key compromise suspected

```bash
# Rotate your API key
bash scripts/rotate_api_key.sh
```

---

## 👥 For Team Members

If someone already set up the project:

```bash
# Option 1: Full init (safe, checks existing)
bash scripts/init_secrets_and_env.sh

# Option 2: Just retrieve existing secrets
bash scripts/setup_local_dev.sh
```

Both work! The init script will just retrieve existing secrets.

---

## 🚀 Full Workflow

### First Time Setup

```bash
# 1. Clone repo
git clone <repo-url>
cd periodicdent42

# 2. Authenticate with GCP
gcloud auth application-default login
gcloud config set project periodicdent42

# 3. Run init script (does everything!)
bash scripts/init_secrets_and_env.sh

# 4. Set up Python environment
cd app
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 5. Start server
uvicorn src.api.main:app --reload --port 8080

# 6. Visit http://localhost:8080/docs
```

**Total time**: ~3-5 minutes

---

## ⚙️ Configuration Options

The generated `.env` file includes:

```bash
# Development mode (auth disabled for easy testing)
ENABLE_AUTH=false

# Your unique API key (from Secret Manager)
API_KEY=<your-generated-key>

# CORS (allows localhost in dev mode)
ALLOWED_ORIGINS=

# Rate limiting
RATE_LIMIT_PER_MINUTE=60

# And more... (see app/.env after running script)
```

You can edit `app/.env` anytime to change these values.

---

## 🔐 Security Features

The script ensures:

- ✅ Secrets use cryptographically secure random generation (`openssl rand`)
- ✅ Files are protected (chmod 600)
- ✅ All files are gitignored (can never be committed)
- ✅ Production uses different secrets (from Secret Manager)
- ✅ No hardcoded secrets anywhere

---

## 🆘 Troubleshooting

### "Not authenticated with gcloud"

```bash
gcloud auth application-default login
```

### "Permission denied" on Secret Manager

```bash
# Make sure you're an Owner/Editor on the project
gcloud projects add-iam-policy-binding periodicdent42 \
  --member="user:YOUR_EMAIL@example.com" \
  --role="roles/owner"
```

### Want to regenerate secrets?

```bash
# Delete existing secrets
gcloud secrets delete api-key --project=periodicdent42

# Run init script again
bash scripts/init_secrets_and_env.sh
# It will generate new ones!
```

---

## 📚 More Information

- **[SECRETS_MANAGEMENT.md](SECRETS_MANAGEMENT.md)** - Complete secrets guide
- **[LOCAL_DEV_SETUP.md](LOCAL_DEV_SETUP.md)** - Detailed local setup
- **[PRODUCTION_DEPLOYMENT_GUIDE.md](PRODUCTION_DEPLOYMENT_GUIDE.md)** - Deploy to Cloud Run

---

## ✅ TL;DR

```bash
bash scripts/init_secrets_and_env.sh  # Do everything
cd app && source venv/bin/activate    # Activate venv
uvicorn src.api.main:app --reload     # Start server
```

**That's it!** 🎉


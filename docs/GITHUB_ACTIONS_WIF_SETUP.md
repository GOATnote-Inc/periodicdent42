# GitHub Actions Workload Identity Federation Setup

**Purpose**: Enable automatic Cloud Run deployments from GitHub Actions without service account keys.

**Current Status**: ❌ Not configured (deployments are manual)  
**Estimated Setup Time**: 15 minutes

---

## Why WIF?

**Without WIF**:
- ❌ Deployments fail with "GCP credentials not configured"
- ❌ Requires manual deployment after each commit
- ❌ Service account key management is risky

**With WIF**:
- ✅ Automatic deployment on push to `main`
- ✅ No service account keys (more secure)
- ✅ GitHub authenticates directly to Google Cloud

---

## Prerequisites

- [x] Google Cloud Project: `periodicdent42`
- [x] GitHub Repository: `GOATnote-Inc/periodicdent42`
- [x] Cloud Run service: `ard-backend`
- [ ] WIF Pool configured
- [ ] GitHub secrets configured

---

## Step 1: Enable Required APIs

```bash
gcloud services enable \
  iamcredentials.googleapis.com \
  cloudresourcemanager.googleapis.com \
  sts.googleapis.com \
  --project=periodicdent42
```

---

## Step 2: Create Workload Identity Pool

```bash
gcloud iam workload-identity-pools create github-pool \
  --location=global \
  --display-name="GitHub Actions Pool" \
  --description="Identity pool for GitHub Actions CI/CD" \
  --project=periodicdent42
```

**Verify**:
```bash
gcloud iam workload-identity-pools describe github-pool \
  --location=global \
  --project=periodicdent42
```

---

## Step 3: Create Workload Identity Provider

```bash
gcloud iam workload-identity-pools providers create-oidc github-provider \
  --location=global \
  --workload-identity-pool=github-pool \
  --issuer-uri="https://token.actions.githubusercontent.com" \
  --attribute-mapping="google.subject=assertion.sub,attribute.actor=assertion.actor,attribute.repository=assertion.repository,attribute.repository_owner=assertion.repository_owner" \
  --attribute-condition="assertion.repository_owner == 'GOATnote-Inc'" \
  --project=periodicdent42
```

**Explanation**:
- `issuer-uri`: GitHub's OIDC token endpoint
- `attribute-mapping`: Maps GitHub token claims to Google Cloud attributes
- `attribute-condition`: Restricts to your organization only

**Verify**:
```bash
gcloud iam workload-identity-pools providers describe github-provider \
  --location=global \
  --workload-identity-pool=github-pool \
  --project=periodicdent42
```

---

## Step 4: Create Service Account for GitHub Actions

```bash
gcloud iam service-accounts create github-actions \
  --display-name="GitHub Actions Deployment" \
  --description="Service account for GitHub Actions to deploy to Cloud Run" \
  --project=periodicdent42
```

**Service Account Email**:
```
github-actions@periodicdent42.iam.gserviceaccount.com
```

---

## Step 5: Grant Required Permissions

```bash
# Cloud Run Admin (deploy services)
gcloud projects add-iam-policy-binding periodicdent42 \
  --member="serviceAccount:github-actions@periodicdent42.iam.gserviceaccount.com" \
  --role="roles/run.admin"

# Service Account User (act as other service accounts)
gcloud projects add-iam-policy-binding periodicdent42 \
  --member="serviceAccount:github-actions@periodicdent42.iam.gserviceaccount.com" \
  --role="roles/iam.serviceAccountUser"

# Storage Admin (upload container images)
gcloud projects add-iam-policy-binding periodicdent42 \
  --member="serviceAccount:github-actions@periodicdent42.iam.gserviceaccount.com" \
  --role="roles/storage.admin"

# Cloud Build Editor (trigger builds)
gcloud projects add-iam-policy-binding periodicdent42 \
  --member="serviceAccount:github-actions@periodicdent42.iam.gserviceaccount.com" \
  --role="roles/cloudbuild.builds.editor"
```

**Verify**:
```bash
gcloud projects get-iam-policy periodicdent42 \
  --flatten="bindings[].members" \
  --filter="bindings.members:github-actions@periodicdent42.iam.gserviceaccount.com" \
  --format="table(bindings.role)"
```

---

## Step 6: Allow GitHub to Impersonate Service Account

```bash
gcloud iam service-accounts add-iam-policy-binding \
  github-actions@periodicdent42.iam.gserviceaccount.com \
  --role="roles/iam.workloadIdentityUser" \
  --member="principalSet://iam.googleapis.com/projects/293837893611/locations/global/workloadIdentityPools/github-pool/attribute.repository/GOATnote-Inc/periodicdent42" \
  --project=periodicdent42
```

**Explanation**: This allows GitHub Actions running in your repository to impersonate the service account.

**Verify**:
```bash
gcloud iam service-accounts get-iam-policy \
  github-actions@periodicdent42.iam.gserviceaccount.com \
  --project=periodicdent42
```

---

## Step 7: Get Workload Identity Provider Resource Name

```bash
gcloud iam workload-identity-pools providers describe github-provider \
  --location=global \
  --workload-identity-pool=github-pool \
  --project=periodicdent42 \
  --format="value(name)"
```

**Output** (save this):
```
projects/293837893611/locations/global/workloadIdentityPools/github-pool/providers/github-provider
```

---

## Step 8: Configure GitHub Secrets

1. **Navigate to GitHub Repository Settings**:
   ```
   https://github.com/GOATnote-Inc/periodicdent42/settings/secrets/actions
   ```

2. **Add Secret: `WIF_PROVIDER`**
   - Click "New repository secret"
   - Name: `WIF_PROVIDER`
   - Value: `projects/293837893611/locations/global/workloadIdentityPools/github-pool/providers/github-provider`
   - Click "Add secret"

3. **Add Secret: `WIF_SERVICE_ACCOUNT`**
   - Click "New repository secret"
   - Name: `WIF_SERVICE_ACCOUNT`
   - Value: `github-actions@periodicdent42.iam.gserviceaccount.com`
   - Click "Add secret"

4. **Add Secret: `GCP_PROJECT_ID`** (if not exists)
   - Click "New repository secret"
   - Name: `GCP_PROJECT_ID`
   - Value: `periodicdent42`
   - Click "Add secret"

---

## Step 9: Verify GitHub Actions Workflow

Your workflow (`.github/workflows/cicd.yaml`) should already have the WIF authentication step:

```yaml
- name: Authenticate to Google Cloud
  if: github.ref == 'refs/heads/main'
  uses: google-github-actions/auth@v2
  with:
    workload_identity_provider: ${{ secrets.WIF_PROVIDER }}
    service_account: ${{ secrets.WIF_SERVICE_ACCOUNT }}

- name: Deploy to Cloud Run
  if: github.ref == 'refs/heads/main'
  run: |
    gcloud run deploy ard-backend \
      --source . \
      --region us-central1 \
      --allow-unauthenticated
```

**Check existing workflow**:
```bash
cat /Users/kiteboard/periodicdent42/.github/workflows/cicd.yaml | grep -A 10 "Authenticate to Google Cloud"
```

---

## Step 10: Test Deployment

### Trigger a Test Deployment

```bash
# Make a trivial change
cd /Users/kiteboard/periodicdent42
echo "# WIF test" >> README.md
git add README.md
git commit -m "test: Trigger WIF deployment test"
git push origin main
```

### Monitor GitHub Actions

1. Go to: https://github.com/GOATnote-Inc/periodicdent42/actions
2. Click on the latest workflow run
3. Expand "Deploy to Cloud Run" step
4. Should see: "Deploying container to Cloud Run service [ard-backend]..."

### Verify Deployment

```bash
# Check latest revision
gcloud run services describe ard-backend --region=us-central1 --format="value(status.latestReadyRevisionName)"

# Test endpoint
curl -s https://ard-backend-dydzexswua-uc.a.run.app/api/htc/health | jq '.status'
```

---

## Troubleshooting

### Issue 1: "Permission denied on workload identity pool"

**Solution**: Verify attribute condition matches your repository:
```bash
gcloud iam workload-identity-pools providers describe github-provider \
  --location=global \
  --workload-identity-pool=github-pool \
  --project=periodicdent42 \
  --format="value(attributeCondition)"
```

Should output:
```
assertion.repository_owner == 'GOATnote-Inc'
```

### Issue 2: "Failed to create token"

**Solution**: Check service account binding:
```bash
gcloud iam service-accounts get-iam-policy \
  github-actions@periodicdent42.iam.gserviceaccount.com \
  --project=periodicdent42
```

Should show `roles/iam.workloadIdentityUser` for the GitHub repository.

### Issue 3: "Deployment still skipped"

**Solution**: Verify secrets are set in GitHub:
```bash
# Check via GitHub CLI (if installed)
gh secret list --repo GOATnote-Inc/periodicdent42

# Or check manually at:
# https://github.com/GOATnote-Inc/periodicdent42/settings/secrets/actions
```

### Issue 4: "Insufficient permissions to deploy"

**Solution**: Verify all required roles are granted:
```bash
gcloud projects get-iam-policy periodicdent42 \
  --flatten="bindings[].members" \
  --filter="bindings.members:github-actions@periodicdent42.iam.gserviceaccount.com" \
  --format="table(bindings.role)"
```

Should show:
- `roles/run.admin`
- `roles/iam.serviceAccountUser`
- `roles/storage.admin`
- `roles/cloudbuild.builds.editor`

---

## Security Best Practices

### 1. Restrict to Specific Repository

Already configured via `attribute-condition`:
```
assertion.repository_owner == 'GOATnote-Inc'
```

To further restrict to specific repository:
```bash
gcloud iam workload-identity-pools providers update-oidc github-provider \
  --location=global \
  --workload-identity-pool=github-pool \
  --attribute-condition="assertion.repository == 'GOATnote-Inc/periodicdent42'" \
  --project=periodicdent42
```

### 2. Restrict to Main Branch Only

Add to workflow:
```yaml
if: github.ref == 'refs/heads/main'
```

### 3. Audit WIF Usage

```bash
gcloud logging read "protoPayload.methodName='GenerateAccessToken' AND protoPayload.serviceData.policyDelta.bindingDeltas.member=~'github-actions@periodicdent42.iam.gserviceaccount.com'" --limit=50
```

### 4. Rotate Service Account Periodically

```bash
# No keys to rotate with WIF! 🎉
# But you can recreate the service account if needed:
# 1. Create new service account
# 2. Grant permissions
# 3. Update WIF binding
# 4. Update GitHub secret
# 5. Delete old service account
```

---

## Cost Implications

**WIF is FREE** ✅

- No additional cost for Workload Identity Federation
- Only pay for Cloud Run deployments (same as manual)
- No Cloud Storage costs for service account keys

---

## Rollback Plan

If WIF causes issues, revert to manual deployment:

1. **Remove secrets from GitHub** (prevents auto-deploy)
2. **Deploy manually**:
   ```bash
   cd /Users/kiteboard/periodicdent42/app
   gcloud builds submit --tag gcr.io/periodicdent42/ard-backend:latest .
   gcloud run deploy ard-backend --image gcr.io/periodicdent42/ard-backend:latest --region us-central1
   ```

---

## Verification Checklist

- [ ] WIF pool created (`github-pool`)
- [ ] WIF provider created (`github-provider`)
- [ ] Service account created (`github-actions`)
- [ ] Permissions granted (4 roles)
- [ ] Service account binding configured
- [ ] GitHub secrets added (2 secrets)
- [ ] Test deployment successful
- [ ] Monitoring configured

---

## Quick Reference

### Get WIF Provider Name
```bash
gcloud iam workload-identity-pools providers describe github-provider \
  --location=global --workload-identity-pool=github-pool \
  --project=periodicdent42 --format="value(name)"
```

### Get Service Account Email
```
github-actions@periodicdent42.iam.gserviceaccount.com
```

### GitHub Secrets Required
1. `WIF_PROVIDER` → Full provider resource name
2. `WIF_SERVICE_ACCOUNT` → Service account email
3. `GCP_PROJECT_ID` → `periodicdent42`

---

## Next Steps After Setup

1. **Monitor first few deployments** closely
2. **Set up Slack notifications** for deployment status
3. **Configure branch protection** (require PR reviews)
4. **Add deployment approval gates** for production

---

**Setup Script**: Run all commands above in sequence  
**Documentation**: https://cloud.google.com/iam/docs/workload-identity-federation  
**Support**: b@thegoatnote.com

---

**Last Updated**: October 10, 2025  
**Status**: 🟡 Pending Setup


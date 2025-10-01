#!/bin/bash
# Security checker - Verify no secrets leaked in git or terminal history

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🔍 Security Integrity Check"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

ISSUES=0

# 1. Check git for committed secrets
echo "1️⃣  Checking git repository for committed secrets..."
if git log --all --full-history -- ".env" ".api-key" "*.env" 2>/dev/null | grep -q "commit"; then
    echo "❌ FAIL: Secret files found in git history!"
    ISSUES=$((ISSUES + 1))
else
    echo "✅ PASS: No secret files in git history"
fi

# 2. Check for .env files tracked by git
echo ""
echo "2️⃣  Checking for tracked .env files..."
if git ls-files --cached | grep -E "\.env$|\.api-key$" >/dev/null 2>&1; then
    echo "❌ FAIL: Secret files are tracked by git!"
    git ls-files --cached | grep -E "\.env$|\.api-key$"
    ISSUES=$((ISSUES + 1))
else
    echo "✅ PASS: No secret files tracked by git"
fi

# 3. Verify .gitignore is protecting secrets
echo ""
echo "3️⃣  Verifying .gitignore protection..."
PROTECTED=0
for file in ".env" ".api-key" ".service-url" "app/.env"; do
    if git check-ignore -q "$file" 2>/dev/null; then
        PROTECTED=$((PROTECTED + 1))
    else
        echo "⚠️  WARNING: $file not ignored by git"
        ISSUES=$((ISSUES + 1))
    fi
done

if [ $PROTECTED -eq 4 ]; then
    echo "✅ PASS: All secret files in .gitignore"
else
    echo "❌ FAIL: Some secret files not in .gitignore ($PROTECTED/4)"
fi

# 4. Check for hardcoded API keys in source code
echo ""
echo "4️⃣  Scanning source code for hardcoded secrets..."
if git grep -i "api[_-]key.*=.*['\"][a-f0-9]{32,}" -- "*.py" "*.js" "*.ts" >/dev/null 2>&1; then
    echo "❌ FAIL: Possible hardcoded API keys found!"
    git grep -n -i "api[_-]key.*=.*['\"][a-f0-9]{32,}" -- "*.py" "*.js" "*.ts" || true
    ISSUES=$((ISSUES + 1))
else
    echo "✅ PASS: No hardcoded API keys in source code"
fi

# 5. Check terminal history for leaked API keys (if .api-key exists)
echo ""
echo "5️⃣  Checking terminal history for leaked secrets..."
if [ -f ".api-key" ]; then
    API_KEY=$(cat .api-key 2>/dev/null | tr -d '\n')
    
    if [ -n "$API_KEY" ]; then
        # Check zsh history
        if [ -f "$HOME/.zsh_history" ]; then
            if grep -q "$API_KEY" "$HOME/.zsh_history" 2>/dev/null; then
                echo "❌ FAIL: API key found in ~/.zsh_history!"
                echo "   Run: echo \"\" > ~/.zsh_history && history -c"
                ISSUES=$((ISSUES + 1))
            fi
        fi
        
        # Check bash history
        if [ -f "$HOME/.bash_history" ]; then
            if grep -q "$API_KEY" "$HOME/.bash_history" 2>/dev/null; then
                echo "❌ FAIL: API key found in ~/.bash_history!"
                echo "   Run: echo \"\" > ~/.bash_history && history -c"
                ISSUES=$((ISSUES + 1))
            fi
        fi
        
        if [ $ISSUES -eq 0 ]; then
            echo "✅ PASS: No API key found in terminal history"
        fi
    else
        echo "⚠️  SKIP: No API key file to check"
    fi
else
    echo "⚠️  SKIP: .api-key file not found"
fi

# 6. Check file permissions on sensitive files
echo ""
echo "6️⃣  Checking file permissions..."
PERM_OK=0
for file in ".api-key" "app/.env"; do
    if [ -f "$file" ]; then
        PERMS=$(stat -f "%A" "$file" 2>/dev/null || stat -c "%a" "$file" 2>/dev/null)
        if [ "$PERMS" != "600" ]; then
            echo "⚠️  WARNING: $file has permissions $PERMS (should be 600)"
            echo "   Run: chmod 600 $file"
            ISSUES=$((ISSUES + 1))
        else
            PERM_OK=$((PERM_OK + 1))
        fi
    fi
done

if [ $PERM_OK -gt 0 ]; then
    echo "✅ PASS: Secret files have secure permissions (600)"
fi

# 7. Check scripts for plain text secret output (to terminal, not files/pipes)
echo ""
echo "7️⃣  Checking scripts for unsafe secret logging..."
# Look for echo commands that print full secrets to terminal (not redirected or piped)
# Exclude: 
#   - echo ... > file (redirected)
#   - echo ... | command (piped)
#   - echo -n ... | command (piped)
#   - echo "...\$API_KEY..." (escaped variable in instructions)
#   - echo "...${VAR:0:8}..." (masked/partial output)
UNSAFE_SCRIPTS=$(grep -r "echo \".*\$[^\\].*API_KEY" scripts/ infra/ 2>/dev/null | \
    grep -v "echo -n\|>\||\|masked\|partial\|hidden\|:0:\|: -\|curl.*-H\|To test" | wc -l | tr -d ' ')

if [ "$UNSAFE_SCRIPTS" -gt 0 ]; then
    echo "⚠️  WARNING: Found scripts that may echo secrets to terminal:"
    grep -rn "echo \".*\$[^\\].*API_KEY" scripts/ infra/ 2>/dev/null | \
        grep -v "echo -n\|>\||\|masked\|partial\|hidden\|:0:\|: -\|curl.*-H\|To test" || true
    echo ""
    echo "   Review these lines to ensure secrets are masked or redirected"
    ISSUES=$((ISSUES + 1))
else
    echo "✅ PASS: Scripts use masked output for secrets"
fi

# Summary
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if [ $ISSUES -eq 0 ]; then
    echo "✅ Security Check: PASSED"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "🔒 Your repository is secure!"
    exit 0
else
    echo "❌ Security Check: FAILED"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo ""
    echo "⚠️  Found $ISSUES issue(s) - please fix before committing!"
    echo ""
    echo "💡 Quick fixes:"
    echo "   • Add missing files to .gitignore"
    echo "   • Remove secrets from git: git rm --cached <file>"
    echo "   • Clear terminal history: history -c && echo \"\" > ~/.zsh_history"
    echo "   • Fix file permissions: chmod 600 .api-key app/.env"
    exit 1
fi


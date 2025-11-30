---
title: Secret Cleanup
slug: reference/secret-cleanup
description: Incident response steps for leaked secrets in NexaCompute.
---

# Secret Cleanup Guide

## 🚨 Critical: Rotate Keys First!

Before cleaning history, **revoke and regenerate all API keys** that may have been exposed:

- **OpenAI**: https://platform.openai.com/api-keys
- **OpenRouter**: https://openrouter.ai/keys  
- **HuggingFace**: https://huggingface.co/settings/tokens

---

## Quick Fix (Automated)

### Step 1: Run Cleanup Script

```bash
./scripts/fix_leaked_secrets.sh
```

This script will:
- Create a backup of your repo
- Remove `.env` from Git history
- Replace API keys with "REMOVED" in history
- Force push cleaned history (with confirmation)

### Step 2: Set Up Protection

```bash
./scripts/setup_secret_protection.sh
```

This installs:
- `detect-secrets` for scanning
- `pre-commit` hooks to block commits with secrets

---

## Manual Commands (Step-by-Step)

### 1. Install git-filter-repo

```bash
pip install git-filter-repo
```

### 2. Remove .env from History

```bash
git filter-repo --path .env --invert-paths --force
```

### 3. Replace API Keys in History

Create a replacement file:

```bash
cat > /tmp/replacements.txt << 'EOF'
sk-[A-Za-z0-9_]{20,}==>REMOVED
sk-or-[A-Za-z0-9_]{20,}==>REMOVED
hf_[A-Za-z0-9_]{20,}==>REMOVED
ghp_[A-Za-z0-9_]{20,}==>REMOVED
EOF

git filter-repo --replace-text /tmp/replacements.txt --force
```

### 4. Force Push Cleaned History

⚠️ **Warning**: This overwrites remote history!

```bash
git push origin --force --all
git push origin --force --tags
```

### 5. Verify Cleanup

```bash
git log --all --full-history -p | grep -E "sk-[A-Za-z0-9_]{20,}" || echo "✅ Clean!"
```

If nothing is found, you're clean!

---

## Contact GitHub Support

After cleanup, open a support ticket:

**URL**: https://support.github.com

**Message**:
> "I recently removed secrets from my private repository history using git-filter-repo. Could you please purge any cached objects containing these secrets from GitHub's internal mirrors to ensure complete cleanup?"

They'll confirm the purge within 24-48 hours.

---

## Prevention Setup

### Enhanced .gitignore

Already updated with comprehensive patterns:

```
.env
.env.*
*.env
*.key
*.pem
*.token
*.secret
```

### Pre-commit Hooks

The `setup_secret_protection.sh` script configures:
- Automatic secret detection
- Blocked commits containing secrets
- Baseline to avoid false positives

### Environment Template

Create `.env.template` (never commit `.env`):

```bash
# .env.template
OPENAI_API_KEY=your_key_here
OPENROUTER_API_KEY=your_key_here
HF_TOKEN=your_token_here
```

---

## Verification Checklist

- [ ] All API keys rotated
- [ ] Cleanup script executed
- [ ] Protection hooks installed
- [ ] `.gitignore` updated
- [ ] GitHub support contacted
- [ ] Team members notified (if applicable)
- [ ] Local `.env` file exists and is ignored

---

## Best Practices Going Forward

1. **Never commit `.env` files** - Always use `.env.template`
2. **Use environment variables** - Load via `python-dotenv` or system env
3. **Pre-commit hooks** - Let automation catch mistakes
4. **Regular rotation** - Rotate keys quarterly or when team changes
5. **CI/CD secrets** - Use GitHub Actions secrets, not files
6. **Key management** - Consider 1Password, Bitwarden, or AWS Secrets Manager

---

## Troubleshooting

### "git-filter-repo not found"

```bash
pip install git-filter-repo
# Or
brew install git-filter-repo  # macOS
```

### "Force push rejected"

You may need to enable force push in GitHub settings, or use:
```bash
git push origin --force --all --no-verify
```

### "Pre-commit hooks not running"

```bash
pre-commit install --hook-type pre-commit
pre-commit install --hook-type pre-push
```

---

## Recovery

If something goes wrong, your backup is at:
```
~/nexa_compute_backup_YYYYMMDD_HHMMSS/
```

You can restore from there or clone a fresh copy from GitHub.


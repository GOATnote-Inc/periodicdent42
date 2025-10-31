# Security Notice

**⚠️ IMPORTANT: Always use your own secure instances; do not expose credentials.**

This repository contains documentation and examples that reference cloud GPU instances. For security:

1. **Never commit** actual IP addresses, ports, or SSH keys
2. **Always use** your own secure cloud instances (RunPod, Vast.ai, Lambda Labs, etc.)
3. **Rotate credentials** regularly if you suspect exposure
4. **Use SSH keys** with passphrase protection
5. **Enable 2FA** on all cloud provider accounts

## Secure Setup Examples

### ✅ Good (Use Placeholders)
```bash
# In documentation
ssh -p [YOUR_SSH_PORT] [YOUR_USER]@[YOUR_INSTANCE_IP]

# In code
H100_IP = os.environ.get("H100_IP", "localhost")
H100_PORT = int(os.environ.get("H100_PORT", "22"))
```

### ❌ Bad (Never Do This)
```bash
# Never hardcode real credentials
ssh -p 25754 root@154.57.34.90  # EXPOSED!
```

## Environment Variables

Always use environment variables for sensitive data:

```bash
# .env (add to .gitignore!)
H100_IP=your.instance.ip
H100_PORT=your_port
SSH_KEY_PATH=/path/to/your/key

# Load in scripts
export H100_IP=$(cat .env | grep H100_IP | cut -d= -f2)
```

## Reporting Security Issues

If you find exposed credentials in this repository:
1. **DO NOT** create a public issue
2. Email: security@yourproject.com (or create private security advisory)
3. We will respond within 24 hours

---

**Remember**: Security is everyone's responsibility. When in doubt, use placeholders.


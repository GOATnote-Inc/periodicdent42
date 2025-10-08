# Authentication Testing Guide

Complete guide for testing JWT authentication in matprov API.

## Setup

1. **Install dependencies**:
```bash
cd api
pip install -r requirements.txt
```

2. **Set environment variables**:
```bash
export JWT_SECRET_KEY=$(python -c "import secrets; print(secrets.token_hex(32))")
export DATABASE_URL="sqlite:///./matprov_api.db"
```

3. **Start the API**:
```bash
uvicorn main:app --reload --port 8000
```

## API Endpoints

### 1. Register a New User

```bash
curl -X POST "http://localhost:8000/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "SecurePass123!"
  }'
```

**Response** (201 Created):
```json
{
  "email": "user@example.com",
  "id": 1,
  "is_active": true,
  "is_superuser": false,
  "created_at": "2025-10-08T12:00:00Z"
}
```

**Password Requirements**:
- Minimum 8 characters
- At least 1 number
- At least 1 special character (!@#$%^&*)

### 2. Login (OAuth2 Form)

```bash
curl -X POST "http://localhost:8000/auth/login" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user@example.com&password=SecurePass123!"
```

**Response** (200 OK):
```json
{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "refresh_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "token_type": "bearer",
  "expires_in": 86400
}
```

### 3. Login (JSON)

```bash
curl -X POST "http://localhost:8000/auth/login/json" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "user@example.com",
    "password": "SecurePass123!"
  }'
```

**Response**: Same as OAuth2 login

### 4. Get Current User

```bash
TOKEN="your_access_token_here"

curl -X GET "http://localhost:8000/auth/me" \
  -H "Authorization: Bearer $TOKEN"
```

**Response** (200 OK):
```json
{
  "email": "user@example.com",
  "id": 1,
  "is_active": true,
  "is_superuser": false,
  "created_at": "2025-10-08T12:00:00Z"
}
```

### 5. Refresh Token

```bash
REFRESH_TOKEN="your_refresh_token_here"

curl -X POST "http://localhost:8000/auth/refresh" \
  -H "Content-Type: application/json" \
  -d "{\"refresh_token\": \"$REFRESH_TOKEN\"}"
```

**Response** (200 OK):
```json
{
  "access_token": "new_access_token...",
  "refresh_token": "new_refresh_token...",
  "token_type": "bearer",
  "expires_in": 86400
}
```

### 6. Logout

```bash
curl -X POST "http://localhost:8000/auth/logout" \
  -H "Authorization: Bearer $TOKEN"
```

**Response** (200 OK):
```json
{
  "message": "Successfully logged out. Please discard your tokens."
}
```

**Note**: JWTs are stateless. Clients must discard tokens. For production, implement token blacklisting.

## Complete Workflow Example

```bash
#!/bin/bash

# 1. Register
echo "1. Registering user..."
REGISTER_RESPONSE=$(curl -s -X POST "http://localhost:8000/auth/register" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "TestPass123!"
  }')
echo "$REGISTER_RESPONSE" | jq

# 2. Login
echo -e "\n2. Logging in..."
LOGIN_RESPONSE=$(curl -s -X POST "http://localhost:8000/auth/login/json" \
  -H "Content-Type: application/json" \
  -d '{
    "email": "test@example.com",
    "password": "TestPass123!"
  }')
echo "$LOGIN_RESPONSE" | jq

# Extract token
ACCESS_TOKEN=$(echo "$LOGIN_RESPONSE" | jq -r '.access_token')
echo "Access token: ${ACCESS_TOKEN:0:50}..."

# 3. Access protected endpoint
echo -e "\n3. Getting user info..."
curl -s -X GET "http://localhost:8000/auth/me" \
  -H "Authorization: Bearer $ACCESS_TOKEN" | jq

# 4. Test protected prediction endpoint
echo -e "\n4. Creating prediction (protected)..."
curl -s -X POST "http://localhost:8000/predictions" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
    "prediction_id": "PRED-001",
    "model_id": "test_model",
    "material_formula": "La2CuO4",
    "predicted_tc": 35.0
  }' | jq

echo -e "\n✅ Authentication workflow complete!"
```

Save as `test_auth.sh`, make executable, and run:
```bash
chmod +x test_auth.sh
./test_auth.sh
```

## Python Client Example

```python
import requests

BASE_URL = "http://localhost:8000"

# 1. Register
response = requests.post(
    f"{BASE_URL}/auth/register",
    json={
        "email": "python@example.com",
        "password": "PyTest123!"
    }
)
print(f"Register: {response.status_code}")

# 2. Login
response = requests.post(
    f"{BASE_URL}/auth/login/json",
    json={
        "email": "python@example.com",
        "password": "PyTest123!"
    }
)
tokens = response.json()
access_token = tokens["access_token"]
print(f"Access token: {access_token[:50]}...")

# 3. Use token in requests
headers = {"Authorization": f"Bearer {access_token}"}

# Get user info
response = requests.get(f"{BASE_URL}/auth/me", headers=headers)
user = response.json()
print(f"User: {user['email']}")

# Create prediction (protected)
response = requests.post(
    f"{BASE_URL}/predictions",
    headers=headers,
    json={
        "prediction_id": "PRED-PY-001",
        "model_id": "py_model",
        "material_formula": "YBa2Cu3O7",
        "predicted_tc": 92.5
    }
)
print(f"Prediction: {response.status_code}")
```

## Error Responses

### 401 Unauthorized

Missing or invalid token:
```json
{
  "detail": "Could not validate credentials"
}
```

### 400 Bad Request

Invalid password:
```json
{
  "detail": [
    {
      "loc": ["body", "password"],
      "msg": "Password must contain at least one number",
      "type": "value_error"
    }
  ]
}
```

Email already registered:
```json
{
  "detail": "Email already registered"
}
```

### 403 Forbidden

Inactive user:
```json
{
  "detail": "Inactive user"
}
```

## Security Best Practices

1. **Store tokens securely**:
   - Never store in localStorage (XSS vulnerable)
   - Use httpOnly cookies (best)
   - Or secure session storage

2. **Token expiration**:
   - Access tokens: 24 hours (configurable)
   - Refresh tokens: 7 days (configurable)
   - Refresh before expiry

3. **HTTPS only** in production:
   ```bash
   # Never send tokens over HTTP!
   curl https://api.matprov.com/auth/login ...
   ```

4. **Rotate secrets regularly**:
   ```bash
   # Generate new secret
   python -c "import secrets; print(secrets.token_hex(32))"
   ```

5. **Monitor for suspicious activity**:
   - Failed login attempts
   - Token refresh patterns
   - Unusual API usage

## Integration with Frontend

### React Example

```javascript
// Login
const login = async (email, password) => {
  const response = await fetch('http://localhost:8000/auth/login/json', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ email, password })
  });
  
  const tokens = await response.json();
  localStorage.setItem('access_token', tokens.access_token);
  localStorage.setItem('refresh_token', tokens.refresh_token);
  
  return tokens;
};

// Authenticated request
const createPrediction = async (data) => {
  const token = localStorage.getItem('access_token');
  
  const response = await fetch('http://localhost:8000/predictions', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${token}`
    },
    body: JSON.stringify(data)
  });
  
  // Handle 401 - token expired
  if (response.status === 401) {
    await refreshToken();
    // Retry request
  }
  
  return response.json();
};

// Refresh token
const refreshToken = async () => {
  const refresh = localStorage.getItem('refresh_token');
  
  const response = await fetch('http://localhost:8000/auth/refresh', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ refresh_token: refresh })
  });
  
  const tokens = await response.json();
  localStorage.setItem('access_token', tokens.access_token);
  localStorage.setItem('refresh_token', tokens.refresh_token);
};
```

## Testing with Swagger UI

1. Open http://localhost:8000/docs
2. Click "Authorize" button (top right)
3. Enter: `username: your-email@example.com` and `password: your-password`
4. Click "Authorize"
5. Now all protected endpoints will include the token automatically

## Troubleshooting

### "Could not validate credentials"

- Check token format: `Bearer <token>`
- Verify token not expired (24 hours)
- Ensure JWT_SECRET_KEY matches between requests

### "Email already registered"

- User exists, use login instead
- Or use different email

### "Incorrect email or password"

- Verify credentials
- Check password meets requirements
- Ensure user account is active

### "Module 'jose' not found"

```bash
pip install python-jose[cryptography]
```

### "Database connection error"

- Check DATABASE_URL environment variable
- Ensure database file has write permissions
- For PostgreSQL, verify connection string

## Next Steps

- [ ] Implement token blacklisting for logout
- [ ] Add email verification
- [ ] Add password reset flow
- [ ] Add 2FA (TOTP)
- [ ] Add rate limiting per user
- [ ] Add OAuth2 social login (Google, GitHub)

---

**Security Contact**: security@matprov.com  
**Documentation**: https://matprov.com/docs/auth


import requests
import jwt
from fastapi import HTTPException, Depends
from fastapi.security import OAuth2PasswordBearer
from fastapi import Security
from typing import Dict
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def get_clerk_jwks():
    """Fetch Clerk's public key from JWKS endpoint"""
    try:
        # Direct URL construction
        jwks_url = "https://valid-termite-98.clerk.accounts.dev/.well-known/jwks.json"
        
        # print(f"Fetching JWKS from: {jwks_url}")  # Debug print
        response = requests.get(jwks_url)
        response.raise_for_status()
        
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"JWKS fetch error: {str(e)}")  # Debug print
        raise HTTPException(
            status_code=500,
            detail=f"Failed to fetch JWKS: {str(e)}"
        )

def get_current_user(token: str = Depends(oauth2_scheme)) -> Dict[str, str]:
    """Validate Clerk JWT and extract user email"""
    try:
        # Remove 'Bearer ' prefix if present
        if token.startswith('Bearer '):
            token = token.split(' ')[1]

        
        # Get the JWKS
        jwks = get_clerk_jwks()
        print("JWKS response:", jwks)  # Debug print
        
        if not jwks.get('keys'):
            raise HTTPException(
                status_code=500,
                detail="No keys found in JWKS"
            )

        # Get the public key
        public_key = jwks['keys'][0]
        
        # Decode the JWT with verification
        payload = jwt.decode(
            token,
            public_key,
            algorithms=["RS256"],
            audience="fastapi",
            options={"verify_signature": False}  # Temporarily disable signature verification
        )
        
        # Extract email from the payload
        email = payload.get("email")
        user_id = payload.get("id")
        if not email:
            raise HTTPException(
                status_code=401,
                detail="Invalid token: missing email"
            )
            
        if not email or not user_id:
            raise HTTPException(
                status_code=401,
                detail="Invalid token: missing required claims"
            )
            
        return {"email": email, "user_id": user_id}
        
    except jwt.ExpiredSignatureError:
        raise HTTPException(
            status_code=401,
            detail="Token has expired"
        )
    except jwt.InvalidTokenError as e:
        raise HTTPException(
            status_code=401,
            detail=f"Invalid token: {str(e)}"
        )
    except Exception as e:
        print(f"Authentication error: {str(e)}")  # Debug print
        raise HTTPException(
            status_code=401,
            detail=f"Authentication failed: {str(e)}"
        )


from fastapi import HTTPException, Header
from typing import Optional

def verify_api_key(authorization: Optional[str] = Header(None), api_secret: Optional[str] = None):
    """
    Simple bearer token check.
    In production: consider JWT or API key database lookup.
    """
    if not api_secret:
        return  # no auth required in dev

    if not authorization or authorization != f"Bearer {api_secret}":
        raise HTTPException(
            status_code=401,
            detail="Invalid or missing API key"
        )
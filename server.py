from fastapi import FastAPI, APIRouter, HTTPException, Request, Response, Depends, Cookie, Header
from fastapi.responses import JSONResponse
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from supabase import create_client, Client
import os, logging, uuid, math, httpx, asyncio, time, base64, bcrypt
from pathlib import Path
from pydantic import BaseModel
from typing import Optional, List
from datetime import datetime, timezone, timedelta
from functools import lru_cache
from PIL import Image
import io
import threading
from collections import defaultdict

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

SUPABASE_URL = os.environ['SUPABASE_URL']
SUPABASE_KEY = os.environ['SUPABASE_SERVICE_ROLE_KEY']

sb: Client = create_client(SUPABASE_URL.rstrip('/'), SUPABASE_KEY)
http_client = httpx.Client(http2=False)
sb.postgrest.session = http_client

app = FastAPI()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- CORS ----------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://campusconnect-app-32.web.app",
        "https://campusconnect-app-32.firebaseapp.com",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
api_router = APIRouter(prefix="/api")

# ---------- Rate limiter with cleanup ----------
RATE_LIMIT_WINDOW = 60
RATE_LIMIT_MAX = 10
rate_limit_store: dict = {}

def check_rate_limit(key: str, max_req: int = RATE_LIMIT_MAX, window: int = RATE_LIMIT_WINDOW):
    now = time.time()
    timestamps = rate_limit_store.get(key, [])
    timestamps = [t for t in timestamps if now - t < window]
    if len(timestamps) >= max_req:
        raise HTTPException(429, "Too many requests. Please slow down.")
    timestamps.append(now)
    rate_limit_store[key] = timestamps

def cleanup_rate_limits():
    while True:
        time.sleep(300)
        now = time.time()
        for key in list(rate_limit_store.keys()):
            timestamps = [t for t in rate_limit_store[key] if now - t < RATE_LIMIT_WINDOW]
            if timestamps:
                rate_limit_store[key] = timestamps
            else:
                del rate_limit_store[key]

threading.Thread(target=cleanup_rate_limits, daemon=True).start()

# ---------- Constants ----------
MAX_GPS_AGE_HOURS = 24
STORAGE_BUCKET = "avatars"
MAX_IMAGE_BASE64_SIZE = 5 * 1024 * 1024

# ---------- Helpers ----------
def _parse_dt(value):
    if value is None: return None
    if isinstance(value, datetime): dt = value
    else:
        s = value.replace("Z", "+00:00") if isinstance(value, str) else value
        dt = datetime.fromisoformat(s)
    if dt.tzinfo is None: dt = dt.replace(tzinfo=timezone.utc)
    return dt

def _maybe(res):
    if res is None: return None
    if hasattr(res, 'error') and res.error:
        logger.error(f"Supabase error: {res.error}")
        return None
    return getattr(res, "data", None)

# Cached reverse geocoding (~1 km granularity)
@lru_cache(maxsize=1024)
def _cached_reverse_geocode(lat_rounded: float, lon_rounded: float) -> tuple:
    try:
        resp = httpx.get(
            "https://nominatim.openstreetmap.org/reverse",
            params={"lat": lat_rounded, "lon": lon_rounded, "format": "json"},
            headers={"User-Agent": "CampusConnect/1.0"},
            timeout=5
        )
        if resp.status_code == 200:
            data = resp.json()
            if data.get("address"):
                country = data["address"].get("country")
                city = data["address"].get("city") or data["address"].get("town") or data["address"].get("village")
                return country, city
    except Exception as e:
        logger.error(f"Reverse geocoding failed: {e}")
    return None, None

def reverse_geocode(lat: float, lon: float) -> tuple:
    return _cached_reverse_geocode(round(lat, 2), round(lon, 2))

async def get_location_from_ip(ip: str) -> dict:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            resp = await client.get(f"http://ip-api.com/json/{ip}?fields=status,lat,lon,country,city")
            if resp.status_code == 200:
                data = resp.json()
                if data.get('status') == 'success':
                    return {
                        'latitude': data.get('lat'),
                        'longitude': data.get('lon'),
                        'country': data.get('country'),
                        'city': data.get('city'),
                        'source': 'ip'
                    }
    except Exception as e:
        logger.error(f"IP geolocation failed: {e}")
    return None

# ---------- Image processing (non‑blocking) ----------
def compress_image_sync(base64_str: str, max_size_kb: int = 300) -> bytes:
    if len(base64_str) > MAX_IMAGE_BASE64_SIZE:
        raise HTTPException(400, "Image too large (max 5 MB)")
    if "," in base64_str:
        base64_str = base64_str.split(",", 1)[1]
    img_data = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_data))
    if img.mode in ("RGBA", "P"):
        img = img.convert("RGB")
    w, h = img.size
    if w > 1200 or h > 1200:
        img.thumbnail((1200, 1200), Image.Resampling.LANCZOS)
    quality = 85
    while True:
        buf = io.BytesIO()
        img.save(buf, format="WEBP", quality=quality)
        size_kb = buf.tell() / 1024
        if size_kb <= max_size_kb or quality <= 20:
            break
        quality -= 5
    return buf.getvalue()

def upload_image_to_supabase_sync(file_bytes: bytes, user_id: str, filename: str) -> str:
    path = f"{user_id}/{filename}"
    sb.storage.from_(STORAGE_BUCKET).upload(
        path=path,
        file=file_bytes,
        file_options={
            "content-type": "image/webp",
            "cache-control": "public, max-age=31536000, immutable"
        }
    )
    return f"{SUPABASE_URL}/storage/v1/object/public/{STORAGE_BUCKET}/{path}"

async def process_image_field_async(image_value: str, user_id: str, filename_prefix: str) -> str:
    if not image_value:
        return image_value
    if image_value.startswith("data:image") or (len(image_value) > 1000 and "base64" in image_value):
        loop = asyncio.get_running_loop()
        compressed = await loop.run_in_executor(None, compress_image_sync, image_value)
        filename = f"{filename_prefix}_{uuid.uuid4().hex[:8]}.webp"
        public_url = await loop.run_in_executor(None, upload_image_to_supabase_sync, compressed, user_id, filename)
        return public_url
    return image_value

# ---------- Models ----------
class LocationUpdatePayload(BaseModel):
    latitude: float
    longitude: float
    accuracy: Optional[float] = None

class GoogleAuthPayload(BaseModel):
    id_token: str
    email: str
    name: str
    picture: str
    ref: Optional[str] = None

class ProfileSetupPayload(BaseModel):
    date_of_birth: str
    display_name: Optional[str] = ""
    gender: str = ""
    year_of_study: str = ""
    course: str = ""
    campus: str = ""
    profile_image: Optional[str] = ""
    gallery_images: Optional[List[str]] = []

class ProfileUpdatePayload(BaseModel):
    date_of_birth: Optional[str] = None
    display_name: Optional[str] = None
    gender: Optional[str] = None
    year_of_study: Optional[str] = None
    course: Optional[str] = None
    campus: Optional[str] = None
    profile_image: Optional[str] = None
    gallery_images: Optional[List[str]] = None

class TutorCreatePayload(BaseModel):
    title: str
    course_name: str
    course_code: str
    price_range: str
    image: Optional[str] = ""

class TutorUpdatePayload(BaseModel):
    title: Optional[str] = None
    course_name: Optional[str] = None
    course_code: Optional[str] = None
    price_range: Optional[str] = None
    image: Optional[str] = None

class TutorReviewPayload(BaseModel):
    rating: int = 0          # 0 = no rating, 1-5 otherwise
    comment: str = ""

class MarketItemCreatePayload(BaseModel):
    title: str
    description: str = ""
    price: str
    category: str
    image: Optional[str] = ""

class MarketItemUpdatePayload(BaseModel):
    title: Optional[str] = None
    description: Optional[str] = None
    price: Optional[str] = None
    category: Optional[str] = None
    image: Optional[str] = None

class MarketMessagePayload(BaseModel):
    content: str

# ---------- Auth ----------
async def get_current_user(
    request: Request,
    session_token_cookie: Optional[str] = Cookie(default=None, alias="session_token"),
    authorization: Optional[str] = Header(default=None),
) -> dict:
    token = session_token_cookie
    if not token and authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1]
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    res = sb.table("user_sessions").select("*").eq("session_token", token).maybe_single().execute()
    session = _maybe(res)
    if not session:
        raise HTTPException(status_code=401, detail="Invalid session")
    if _parse_dt(session["expires_at"]) < datetime.now(timezone.utc):
        raise HTTPException(status_code=401, detail="Session expired")
    user = _maybe(sb.table("users").select("*").eq("user_id", session["user_id"]).maybe_single().execute())
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    if user.get("deleted") or user.get("banned"):
        raise HTTPException(status_code=401, detail="Account deleted or banned")
    sb.table("users").update({"last_active": datetime.now(timezone.utc).isoformat()}).eq("user_id", user["user_id"]).execute()
    return user

# ---------- Root ----------
@app.get("/")
def root():
    return {"message": "CampusConnect API is running"}

@api_router.get("/")
def api_root():
    return {"message": "CampusConnect API v1"}

# ---------- Authentication ----------
@api_router.post("/auth/google")
def auth_google(payload: GoogleAuthPayload, request: Request, response: Response):
    check_rate_limit(f"auth_{request.client.host}", max_req=5)
    email, name, picture = payload.email, payload.name, payload.picture
    session_token = f"session_{uuid.uuid4().hex[:32]}"
    now_iso = datetime.now(timezone.utc).isoformat()

    existing = _maybe(sb.table("users").select("*").eq("email", email).eq("deleted", False).maybe_single().execute())
    if existing:
        user_id = existing["user_id"]
        sb.table("users").update({
            "name": name, "picture": picture, "last_active": now_iso, "deleted": False
        }).eq("user_id", user_id).execute()
    else:
        deleted_user = _maybe(sb.table("users").select("*").eq("email", email).eq("deleted", True).maybe_single().execute())
        if deleted_user:
            if deleted_user.get("banned"):
                raise HTTPException(403, "Your account has been banned.")
            user_id = deleted_user["user_id"]
            sb.table("users").update({
                "name": name, "picture": picture, "last_active": now_iso, "deleted": False
            }).eq("user_id", user_id).execute()
        else:
            user_id = f"user_{uuid.uuid4().hex[:12]}"
            sb.table("users").insert({
                "user_id": user_id, "email": email, "name": name, "picture": picture,
                "created_at": now_iso, "last_active": now_iso,
            }).execute()

    expires_at = datetime.now(timezone.utc) + timedelta(days=7)
    sb.table("user_sessions").upsert({
        "session_token": session_token,
        "user_id": user_id,
        "expires_at": expires_at.isoformat(),
        "created_at": now_iso
    }).execute()

    response.set_cookie(
        key="session_token", value=session_token,
        httponly=True, secure=request.url.scheme == "https",
        samesite="lax", path="/", max_age=7*24*60*60
    )
    return {"ok": True, "user_id": user_id, "token": session_token}

@api_router.get("/auth/me")
def auth_me(user: dict = Depends(get_current_user)):
    profile = _maybe(sb.table("user_profiles").select("*").eq("user_id", user["user_id"]).maybe_single().execute())
    onboarding_complete = profile.get("onboarding_complete", False) if profile else False
    has_gps = profile and profile.get("gps_latitude") is not None
    gps_stale = False
    if has_gps and profile.get("gps_verified_at"):
        gps_age = datetime.now(timezone.utc) - _parse_dt(profile["gps_verified_at"])
        gps_stale = gps_age > timedelta(hours=MAX_GPS_AGE_HOURS)

    return {
        "user_id": user["user_id"],
        "email": user["email"],
        "name": user.get("name"),
        "picture": user.get("picture"),
        "onboarding_complete": onboarding_complete,
        "has_gps": has_gps,
        "gps_stale": gps_stale,
        "needs_location": not has_gps or gps_stale,
        "created_at": user.get("created_at"),
    }

@api_router.post("/accept-privacy")
def accept_privacy(user: dict = Depends(get_current_user)):
    now = datetime.now(timezone.utc).isoformat()
    sb.table("users").update({"privacy_accepted_at": now}).eq("user_id", user["user_id"]).execute()
    return {"ok": True}

@api_router.post("/auth/logout")
def auth_logout(
    response: Response,
    session_token_cookie: Optional[str] = Cookie(default=None, alias="session_token"),
    authorization: Optional[str] = Header(default=None)
):
    token = session_token_cookie
    if not token and authorization and authorization.startswith("Bearer "):
        token = authorization.split(" ", 1)[1]
    if token:
        sb.table("user_sessions").delete().eq("session_token", token).execute()
    response.delete_cookie(key="session_token", path="/", samesite="lax", secure=False)
    return {"ok": True}

@api_router.delete("/auth/me")
async def delete_account(user: dict = Depends(get_current_user)):
    sb.table("users").update({"deleted": True}).eq("user_id", user["user_id"]).execute()
    sb.table("user_sessions").delete().eq("user_id", user["user_id"]).execute()
    sb.table("user_profiles").update({
        "display_name": None,
        "bio": None,
        "date_of_birth": None,
        "profile_image": None,
        "gallery_images": None,
        "gps_latitude": None,
        "gps_longitude": None,
        "gps_verified_at": None,
    }).eq("user_id", user["user_id"]).execute()
    return {"ok": True, "message": "Account deleted"}

# ---------- Email / Password Auth ----------
def send_email(to_email: str, subject: str, body: str):
    brevo_api_key = os.environ.get("BREVO_API_KEY")
    if not brevo_api_key:
        logger.info(f"Email not sent (no API key): {to_email}")
        return
    payload = {
        "sender": {"name": "CampusConnect", "email": "noreply@campusconnect.com"},
        "to": [{"email": to_email}],
        "subject": subject,
        "htmlContent": body,
    }
    try:
        resp = httpx.post(
            "https://api.brevo.com/v3/smtp/email",
            json=payload,
            headers={"api-key": brevo_api_key, "Content-Type": "application/json"},
            timeout=10
        )
        if resp.status_code not in (200, 201, 202):
            logger.error(f"Brevo error {resp.status_code}: {resp.text}")
    except Exception as e:
        logger.error(f"Email send failed: {e}")

@api_router.post("/auth/signup")
def signup_email(payload: dict, request: Request):
    email = payload.get("email", "").strip().lower()
    password = payload.get("password", "")
    name = payload.get("name", email.split("@")[0])
    if not email or not password:
        raise HTTPException(400, "Email and password required")
    if len(password) < 6:
        raise HTTPException(400, "Password must be at least 6 characters")
    if len(password) > 72:
        raise HTTPException(400, "Password too long")
    existing = _maybe(sb.table("users").select("user_id,deleted").eq("email", email).maybe_single().execute())
    if existing:
        if existing.get("deleted"):
            raise HTTPException(400, "Account with this email was deleted. Contact support.")
        raise HTTPException(400, "Email already registered")

    password_hash = bcrypt.hashpw(password[:72].encode("utf-8"), bcrypt.gensalt()).decode()
    user_id = f"user_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).isoformat()
    verification_token = uuid.uuid4().hex
    verification_expires = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()

    sb.table("users").insert({
        "user_id": user_id, "email": email, "name": name,
        "password_hash": password_hash, "email_verified": False,
        "verification_token": verification_token,
        "verification_token_expires": verification_expires,
        "created_at": now, "last_active": now,
    }).execute()

    verify_link = f"https://jwdate-e6fe5.web.app/verify-email?token={verification_token}"
    body = f"<h2>Welcome to CampusConnect!</h2><p>Click to verify: <a href='{verify_link}'>{verify_link}</a></p>"
    threading.Thread(target=send_email, args=(email, "Verify your email", body)).start()
    return {"ok": True, "message": "Account created. Check your email."}

@api_router.post("/auth/login")
def login_email(payload: dict, request: Request, response: Response):
    email = payload.get("email", "").strip().lower()
    password = payload.get("password", "")
    if not email or not password:
        raise HTTPException(400, "Email and password required")
    user = _maybe(sb.table("users").select("*").eq("email", email).eq("deleted", False).maybe_single().execute())
    if not user or not user.get("password_hash"):
        raise HTTPException(401, "Invalid credentials")
    if not bcrypt.checkpw(password[:72].encode("utf-8"), user["password_hash"].encode()):
        raise HTTPException(401, "Invalid credentials")
    if not user.get("email_verified"):
        raise HTTPException(403, "Email not verified. Check your inbox.")

    session_token = f"session_{uuid.uuid4().hex[:32]}"
    expires_at = datetime.now(timezone.utc) + timedelta(days=7)
    sb.table("user_sessions").upsert({
        "session_token": session_token, "user_id": user["user_id"],
        "expires_at": expires_at.isoformat(), "created_at": datetime.now(timezone.utc).isoformat()
    }).execute()
    sb.table("users").update({"last_active": datetime.now(timezone.utc).isoformat()}).eq("user_id", user["user_id"]).execute()

    response.set_cookie(
        key="session_token", value=session_token,
        httponly=True, secure=request.url.scheme == "https",
        samesite="lax", path="/", max_age=7*24*60*60
    )
    return {"ok": True, "user_id": user["user_id"], "token": session_token}

@api_router.post("/auth/verify-email")
def verify_email(payload: dict):
    token = payload.get("token", "")
    user = _maybe(sb.table("users").select("*").eq("verification_token", token).maybe_single().execute())
    if not user:
        raise HTTPException(400, "Invalid or expired token")
    if user.get("email_verified"):
        return {"ok": True, "message": "Already verified"}
    if _parse_dt(user["verification_token_expires"]) < datetime.now(timezone.utc):
        raise HTTPException(400, "Token expired")
    sb.table("users").update({
        "email_verified": True, "verification_token": None, "verification_token_expires": None
    }).eq("user_id", user["user_id"]).execute()
    return {"ok": True, "message": "Email verified!"}

@api_router.post("/auth/forgot-password")
def forgot_password(payload: dict):
    email = payload.get("email", "").strip().lower()
    user = _maybe(sb.table("users").select("*").eq("email", email).eq("deleted", False).maybe_single().execute())
    if user:
        reset_token = uuid.uuid4().hex
        reset_expires = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
        sb.table("users").update({
            "reset_token": reset_token, "reset_token_expires": reset_expires
        }).eq("user_id", user["user_id"]).execute()
        reset_link = f"https://jwdate-e6fe5.web.app/reset-password?token={reset_token}"
        body = f"<h2>Password Reset</h2><p>Click to reset: <a href='{reset_link}'>{reset_link}</a></p>"
        threading.Thread(target=send_email, args=(email, "Reset your password", body)).start()
    return {"ok": True, "message": "If registered, a reset link has been sent."}

@api_router.post("/auth/reset-password")
def reset_password(payload: dict):
    token = payload.get("token", "")
    new_password = payload.get("password", "")
    if not token or not new_password:
        raise HTTPException(400, "Token and new password required")
    if len(new_password) < 6:
        raise HTTPException(400, "Password too short")
    user = _maybe(sb.table("users").select("*").eq("reset_token", token).maybe_single().execute())
    if not user or _parse_dt(user["reset_token_expires"]) < datetime.now(timezone.utc):
        raise HTTPException(400, "Invalid or expired token")
    new_hash = bcrypt.hashpw(new_password[:72].encode("utf-8"), bcrypt.gensalt()).decode()
    sb.table("users").update({
        "password_hash": new_hash, "reset_token": None, "reset_token_expires": None
    }).eq("user_id", user["user_id"]).execute()
    return {"ok": True, "message": "Password reset. You can now log in."}

# ---------- Location ----------
@api_router.post("/location/update")
async def update_location(payload: LocationUpdatePayload, user: dict = Depends(get_current_user)):
    if not (-90 <= payload.latitude <= 90) or not (-180 <= payload.longitude <= 180):
        raise HTTPException(400, "Invalid coordinates")
    if payload.accuracy and payload.accuracy > 500:
        raise HTTPException(400, "Location accuracy too low (>500m).")
    now = datetime.now(timezone.utc)
    country, city = reverse_geocode(payload.latitude, payload.longitude)

    profile_data = {
        "gps_latitude": payload.latitude,
        "gps_longitude": payload.longitude,
        "gps_verified_at": now.isoformat(),
        "gps_accuracy": payload.accuracy,
        "location_source": "gps",
        "latitude": payload.latitude,
        "longitude": payload.longitude,
        "country": country,
        "city": city or "",
        "updated_at": now.isoformat(),
    }

    existing = _maybe(sb.table("user_profiles").select("user_id").eq("user_id", user["user_id"]).maybe_single().execute())
    if existing:
        sb.table("user_profiles").update(profile_data).eq("user_id", user["user_id"]).execute()
    else:
        profile_data["user_id"] = user["user_id"]
        profile_data["created_at"] = now.isoformat()
        sb.table("user_profiles").insert(profile_data).execute()

    return {"ok": True, "latitude": payload.latitude, "longitude": payload.longitude, "country": country, "city": city}

@api_router.get("/location/ip-fallback")
async def ip_fallback(request: Request, user: dict = Depends(get_current_user)):
    client_ip = request.headers.get("x-forwarded-for", request.client.host).split(",")[0].strip()
    location = await get_location_from_ip(client_ip)
    if location:
        now = datetime.now(timezone.utc)
        profile_data = {
            "gps_latitude": location['latitude'],
            "gps_longitude": location['longitude'],
            "gps_verified_at": now.isoformat(),
            "location_source": "ip",
            "latitude": location['latitude'],
            "longitude": location['longitude'],
            "country": location.get('country', ''),
            "city": location.get('city', ''),
            "updated_at": now.isoformat(),
        }
        existing = _maybe(sb.table("user_profiles").select("user_id").eq("user_id", user["user_id"]).maybe_single().execute())
        if existing:
            sb.table("user_profiles").update(profile_data).eq("user_id", user["user_id"]).execute()
        else:
            profile_data["user_id"] = user["user_id"]
            sb.table("user_profiles").insert(profile_data).execute()
        return {"ok": True, "latitude": location['latitude'], "longitude": location['longitude'],
                "country": location.get('country'), "city": location.get('city'), "source": "ip"}
    return {"ok": False, "message": "Could not determine location from IP"}

@api_router.get("/location/status")
def get_location_status(user: dict = Depends(get_current_user)):
    profile = _maybe(sb.table("user_profiles").select("gps_latitude,gps_longitude,gps_verified_at,location_source").eq("user_id", user["user_id"]).maybe_single().execute())
    if not profile or profile.get("gps_latitude") is None:
        return {"has_location": False, "needs_location": True}
    gps_age = datetime.now(timezone.utc) - _parse_dt(profile["gps_verified_at"])
    is_stale = gps_age > timedelta(hours=MAX_GPS_AGE_HOURS)
    return {"has_location": True, "needs_location": is_stale, "is_stale": is_stale, "last_updated": profile.get("gps_verified_at")}

# ---------- Profile ----------
def get_profile(user: dict) -> dict:
    profile = _maybe(sb.table("user_profiles").select("*").eq("user_id", user["user_id"]).maybe_single().execute())
    base = {
        "user_id": user["user_id"],
        "email": user.get("email"),
        "name": user.get("name"),
        "picture": user.get("picture"),
    }
    if not profile:
        return {
            **base,
            "onboarding_complete": False,
            "country": None, "city": None,
            "date_of_birth": None,
            "display_name": user.get("name", ""),
            "gender": "",
            "year_of_study": "",
            "course": "",
            "campus": "",
            "profile_image": user.get("picture", ""),
            "gallery_images": [],
            "location_source": "none"
        }
    return {
        **base,
        "date_of_birth": profile.get("date_of_birth"),
        "display_name": profile.get("display_name", user.get("name", "")),
        "gender": profile.get("gender", ""),
        "year_of_study": profile.get("year_of_study", ""),
        "course": profile.get("course", ""),
        "campus": profile.get("campus", ""),
        "profile_image": profile.get("profile_image") or user.get("picture", ""),
        "gallery_images": profile.get("gallery_images") or [],
        "country": profile.get("country"),
        "city": profile.get("city"),
        "latitude": profile.get("gps_latitude"),
        "longitude": profile.get("gps_longitude"),
        "location_source": profile.get("location_source", "none"),
        "onboarding_complete": profile.get("onboarding_complete", False),
    }

@api_router.post("/profile/setup")
async def setup_profile(payload: ProfileSetupPayload, user: dict = Depends(get_current_user)):
    if not payload.gender.strip() or not payload.year_of_study.strip() or not payload.course.strip() or not payload.campus.strip():
        raise HTTPException(400, "Gender, year of study, course, and campus are required")

    profile_image = await process_image_field_async(payload.profile_image, user["user_id"], "profile")
    gallery = []
    for i, img in enumerate(payload.gallery_images or []):
        gallery.append(await process_image_field_async(img, user["user_id"], f"gallery_{i}"))

    required_fields_present = bool(
        payload.date_of_birth and payload.gender.strip() and
        payload.year_of_study.strip() and payload.course.strip() and payload.campus.strip()
    )

    profile_data = {
        "user_id": user["user_id"],
        "date_of_birth": payload.date_of_birth,
        "display_name": payload.display_name or user.get("name", ""),
        "gender": payload.gender,
        "year_of_study": payload.year_of_study,
        "course": payload.course,
        "campus": payload.campus,
        "profile_image": profile_image,
        "gallery_images": gallery,
        "onboarding_complete": required_fields_present,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

    existing = _maybe(sb.table("user_profiles").select("user_id").eq("user_id", user["user_id"]).maybe_single().execute())
    if existing:
        if existing.get("gps_latitude"):
            profile_data["gps_latitude"] = existing["gps_latitude"]
            profile_data["gps_longitude"] = existing["gps_longitude"]
            profile_data["gps_verified_at"] = existing["gps_verified_at"]
            profile_data["location_source"] = existing.get("location_source", "none")
        sb.table("user_profiles").update(profile_data).eq("user_id", user["user_id"]).execute()
    else:
        profile_data["created_at"] = datetime.now(timezone.utc).isoformat()
        sb.table("user_profiles").insert(profile_data).execute()

    return {"ok": True, "profile": get_profile(user)}

@api_router.put("/profile")
async def update_profile(payload: ProfileUpdatePayload, user: dict = Depends(get_current_user)):
    updates = {}
    for field in ["date_of_birth", "display_name", "gender", "year_of_study", "course", "campus"]:
        if getattr(payload, field, None) is not None:
            updates[field] = getattr(payload, field)

    if payload.profile_image is not None:
        updates["profile_image"] = await process_image_field_async(payload.profile_image, user["user_id"], "profile")
    if payload.gallery_images is not None:
        new_gallery = []
        for i, img in enumerate(payload.gallery_images):
            new_gallery.append(await process_image_field_async(img, user["user_id"], f"gallery_{i}"))
        updates["gallery_images"] = new_gallery

    if not updates:
        return {"ok": True, "profile": get_profile(user)}

    existing = _maybe(sb.table("user_profiles").select("*").eq("user_id", user["user_id"]).maybe_single().execute())
    current_profile = existing if existing else {}
    for k in ["date_of_birth", "gender", "year_of_study", "course", "campus"]:
        if k in updates:
            current_profile[k] = updates[k]
        else:
            current_profile[k] = current_profile.get(k, "")

    required_fields_present = bool(
        current_profile.get("date_of_birth") and current_profile.get("gender") and
        current_profile.get("year_of_study") and current_profile.get("course") and current_profile.get("campus")
    )
    if required_fields_present and not current_profile.get("onboarding_complete"):
        updates["onboarding_complete"] = True

    updates["updated_at"] = datetime.now(timezone.utc).isoformat()
    if existing:
        sb.table("user_profiles").update(updates).eq("user_id", user["user_id"]).execute()
    else:
        updates["user_id"] = user["user_id"]
        updates["onboarding_complete"] = required_fields_present
        updates["created_at"] = datetime.now(timezone.utc).isoformat()
        sb.table("user_profiles").insert(updates).execute()

    return {"ok": True, "profile": get_profile(user)}

@api_router.get("/profile")
def get_my_profile(user: dict = Depends(get_current_user)):
    return get_profile(user)

# ---------- Tutors ----------
@api_router.post("/tutors")
async def create_tutor(payload: TutorCreatePayload, user: dict = Depends(get_current_user)):
    if not payload.title.strip() or not payload.course_name.strip() or not payload.course_code.strip() or not payload.price_range.strip():
        raise HTTPException(400, "All fields are required")
    
    image_url = ""
    if payload.image:
        image_url = await process_image_field_async(payload.image, user["user_id"], "tutor")
    
    tutor_id = f"tutor_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).isoformat()
    
    sb.table("tutors").insert({
        "tutor_id": tutor_id,
        "user_id": user["user_id"],
        "title": payload.title.strip(),
        "course_name": payload.course_name.strip(),
        "course_code": payload.course_code.strip().upper(),
        "price_range": payload.price_range.strip(),
        "image": image_url,
        "created_at": now,
        "updated_at": now
    }).execute()
    
    return {"ok": True, "tutor_id": tutor_id}

@api_router.get("/tutors")
def list_tutors(search: Optional[str] = None):
    query = sb.table("tutors").select("*").order("created_at", desc=True)
    if search:
        query = query.ilike("course_code", f"%{search}%")
    tutors = query.execute().data or []

    # Bulk compute average ratings
    if tutors:
        tutor_ids = [t["tutor_id"] for t in tutors]
        ratings = sb.table("tutor_reviews") \
            .select("tutor_id, rating") \
            .in_("tutor_id", tutor_ids) \
            .execute().data or []

        rating_sums = defaultdict(int)
        rating_counts = defaultdict(int)
        for r in ratings:
            if r["rating"] > 0:
                rating_sums[r["tutor_id"]] += r["rating"]
                rating_counts[r["tutor_id"]] += 1

        for t in tutors:
            tid = t["tutor_id"]
            cnt = rating_counts[tid]
            t["average_rating"] = round(rating_sums[tid] / cnt, 1) if cnt else 0
            t["rating_count"] = cnt
    else:
        for t in tutors:
            t["average_rating"] = 0
            t["rating_count"] = 0

    return tutors

@api_router.get("/tutors/{tutor_id}")
def get_tutor(tutor_id: str):
    tutor = _maybe(sb.table("tutors").select("*").eq("tutor_id", tutor_id).maybe_single().execute())
    if not tutor:
        raise HTTPException(404, "Tutor not found")
    # Calculate average rating
    rating_data = sb.table("tutor_reviews") \
        .select("rating") \
        .eq("tutor_id", tutor_id) \
        .execute().data or []
    valid_ratings = [r["rating"] for r in rating_data if r["rating"] > 0]
    avg_rating = round(sum(valid_ratings) / len(valid_ratings), 1) if valid_ratings else 0
    tutor["average_rating"] = avg_rating
    tutor["rating_count"] = len(valid_ratings)
    return tutor

@api_router.put("/tutors/{tutor_id}")
async def update_tutor(tutor_id: str, payload: TutorUpdatePayload, user: dict = Depends(get_current_user)):
    tutor = _maybe(sb.table("tutors").select("*").eq("tutor_id", tutor_id).maybe_single().execute())
    if not tutor:
        raise HTTPException(404, "Tutor not found")
    if tutor["user_id"] != user["user_id"]:
        raise HTTPException(403, "You can only edit your own ads")
    
    updates = {}
    for field in ["title", "course_name", "course_code", "price_range"]:
        if getattr(payload, field, None) is not None:
            val = getattr(payload, field).strip()
            if field == "course_code":
                val = val.upper()
            updates[field] = val

    if payload.image is not None:
        updates["image"] = await process_image_field_async(payload.image, user["user_id"], "tutor")
    
    if updates:
        updates["updated_at"] = datetime.now(timezone.utc).isoformat()
        sb.table("tutors").update(updates).eq("tutor_id", tutor_id).execute()
    
    return {"ok": True}

@api_router.delete("/tutors/{tutor_id}")
def delete_tutor(tutor_id: str, user: dict = Depends(get_current_user)):
    tutor = _maybe(sb.table("tutors").select("*").eq("tutor_id", tutor_id).maybe_single().execute())
    if not tutor:
        raise HTTPException(404, "Tutor not found")
    if tutor["user_id"] != user["user_id"] and not user.get("is_admin"):
        raise HTTPException(403, "You can only delete your own ads")
    sb.table("tutors").delete().eq("tutor_id", tutor_id).execute()
    return {"ok": True}

# ---------- Tutor Reviews ----------
@api_router.post("/tutors/{tutor_id}/reviews")
def create_tutor_review(tutor_id: str, payload: TutorReviewPayload, user: dict = Depends(get_current_user)):
    tutor = _maybe(sb.table("tutors").select("tutor_id").eq("tutor_id", tutor_id).maybe_single().execute())
    if not tutor:
        raise HTTPException(404, "Tutor not found")
    rating = payload.rating
    if rating > 0 and (rating < 1 or rating > 5):
        raise HTTPException(400, "Rating must be between 1 and 5 if provided")

    if rating > 0:
        # Rating: one per user → update if exists
        existing_rating = _maybe(sb.table("tutor_reviews")
            .select("review_id")
            .eq("tutor_id", tutor_id)
            .eq("user_id", user["user_id"])
            .gt("rating", 0)
            .maybe_single().execute())

        if existing_rating:
            # Update existing rating
            sb.table("tutor_reviews") \
                .update({"rating": rating, "created_at": datetime.now(timezone.utc).isoformat()}) \
                .eq("review_id", existing_rating["review_id"]) \
                .execute()
            return {"ok": True, "review_id": existing_rating["review_id"], "updated": True}
        else:
            # Insert new rating row (comment empty)
            review_id = f"rev_{uuid.uuid4().hex[:12]}"
            now = datetime.now(timezone.utc).isoformat()
            sb.table("tutor_reviews").insert({
                "review_id": review_id,
                "tutor_id": tutor_id,
                "user_id": user["user_id"],
                "rating": rating,
                "comment": "",
                "created_at": now
            }).execute()
            return {"ok": True, "review_id": review_id}
    else:
        # Chat message: always insert new row
        review_id = f"rev_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()
        sb.table("tutor_reviews").insert({
            "review_id": review_id,
            "tutor_id": tutor_id,
            "user_id": user["user_id"],
            "rating": 0,
            "comment": payload.comment or "",
            "created_at": now
        }).execute()
        return {"ok": True, "review_id": review_id}

@api_router.get("/tutors/{tutor_id}/reviews")
def list_tutor_reviews(tutor_id: str):
    reviews = sb.table("tutor_reviews")\
        .select("review_id, tutor_id, user_id, rating, comment, created_at, users!fk_review_user(name, picture)")\
        .eq("tutor_id", tutor_id)\
        .order("created_at", desc=True)\
        .execute().data or []
    enriched = []
    for r in reviews:
        user = r.get("users", {})
        enriched.append({
            "review_id": r["review_id"],
            "tutor_id": r["tutor_id"],
            "user_id": r["user_id"],
            "rating": r["rating"],
            "comment": r["comment"],
            "created_at": r["created_at"],
            "user_name": user.get("name", "Unknown"),
            "user_picture": user.get("picture", ""),
        })
    return enriched


@api_router.get("/tutors/{tutor_id}/my-rating")
def get_my_rating(tutor_id: str, user: dict = Depends(get_current_user)):
    review = _maybe(sb.table("tutor_reviews")
        .select("rating")
        .eq("tutor_id", tutor_id)
        .eq("user_id", user["user_id"])
        .gt("rating", 0)
        .maybe_single().execute())
    return {"rating": review["rating"] if review else 0}


# ---------- My Ads Count ----------
@api_router.get("/tutors/myads/count")
def my_tutor_ads_count(user: dict = Depends(get_current_user)):
    res = sb.table("tutors").select("tutor_id", count="exact").eq("user_id", user["user_id"]).execute()
    count = res.count if hasattr(res, 'count') else 0
    return {"count": count}

# ---------- Marketplace ----------
MARKET_CATEGORIES = ["Products", "Services"]

@api_router.get("/marketplace/categories")
def get_market_categories():
    return MARKET_CATEGORIES

@api_router.post("/marketplace/items")
async def create_market_item(payload: MarketItemCreatePayload, user: dict = Depends(get_current_user)):
    if not payload.title.strip() or not payload.price.strip() or not payload.category.strip():
        raise HTTPException(400, "Title, price, and category are required")
    if payload.category not in MARKET_CATEGORIES:
        raise HTTPException(400, "Invalid category")

    image_url = ""
    if payload.image:
        image_url = await process_image_field_async(payload.image, user["user_id"], "market")

    item_id = f"item_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).isoformat()

    sb.table("marketplace_items").insert({
        "item_id": item_id,
        "user_id": user["user_id"],
        "title": payload.title.strip(),
        "description": payload.description.strip(),
        "price": payload.price.strip(),
        "category": payload.category.strip(),
        "image": image_url,
        "created_at": now,
        "updated_at": now
    }).execute()

    return {"ok": True, "item_id": item_id}

@api_router.get("/marketplace/items")
def list_market_items(category: Optional[str] = None):
    query = sb.table("marketplace_items").select("*").order("created_at", desc=True)
    if category:
        query = query.eq("category", category)
    items = query.execute().data or []

    # --- Add seller info from user_profiles ---
    if items:
        seller_ids = list({item["user_id"] for item in items})
        profiles = sb.table("user_profiles") \
            .select("user_id, display_name, phone_number, country, city") \
            .in_("user_id", seller_ids) \
            .execute().data or []
        profile_map = {p["user_id"]: p for p in profiles}
        for item in items:
            p = profile_map.get(item["user_id"], {})
            item["seller_name"] = p.get("display_name") or "Unknown"
            item["seller_phone"] = p.get("phone_number") or ""
            item["seller_country"] = p.get("country") or ""
            item["seller_city"] = p.get("city") or ""

    return items

@api_router.get("/marketplace/items/{item_id}")
def get_market_item(item_id: str):
    item = _maybe(sb.table("marketplace_items").select("*").eq("item_id", item_id).maybe_single().execute())
    if not item:
        raise HTTPException(404, "Item not found")
    return item

@api_router.put("/marketplace/items/{item_id}")
async def update_market_item(item_id: str, payload: MarketItemUpdatePayload, user: dict = Depends(get_current_user)):
    item = _maybe(sb.table("marketplace_items").select("*").eq("item_id", item_id).maybe_single().execute())
    if not item:
        raise HTTPException(404, "Item not found")
    if item["user_id"] != user["user_id"] and not user.get("is_admin"):
        raise HTTPException(403, "You can only edit your own listings")

    updates = {}
    for field in ["title", "description", "price", "category"]:
        if getattr(payload, field, None) is not None:
            val = getattr(payload, field).strip()
            if field == "category" and val not in MARKET_CATEGORIES:
                raise HTTPException(400, "Invalid category")
            updates[field] = val

    if payload.image is not None:
        updates["image"] = await process_image_field_async(payload.image, user["user_id"], "market")

    if updates:
        updates["updated_at"] = datetime.now(timezone.utc).isoformat()
        sb.table("marketplace_items").update(updates).eq("item_id", item_id).execute()

    return {"ok": True}

@api_router.delete("/marketplace/items/{item_id}")
def delete_market_item(item_id: str, user: dict = Depends(get_current_user)):
    item = _maybe(sb.table("marketplace_items").select("*").eq("item_id", item_id).maybe_single().execute())
    if not item:
        raise HTTPException(404, "Item not found")
    if item["user_id"] != user["user_id"] and not user.get("is_admin"):
        raise HTTPException(403, "You can only delete your own listings")
    sb.table("marketplace_items").delete().eq("item_id", item_id).execute()
    return {"ok": True}

@api_router.get("/marketplace/myitems/count")
def my_market_items_count(user: dict = Depends(get_current_user)):
    res = sb.table("marketplace_items").select("item_id", count="exact").eq("user_id", user["user_id"]).execute()
    count = res.count if hasattr(res, 'count') else 0
    return {"count": count}

# ---------- Marketplace Private Chat ----------
@api_router.post("/marketplace/items/{item_id}/messages")
def send_market_message(item_id: str, payload: MarketMessagePayload, user: dict = Depends(get_current_user)):
    item = _maybe(sb.table("marketplace_items").select("*").eq("item_id", item_id).maybe_single().execute())
    if not item:
        raise HTTPException(404, "Item not found")
    receiver_id = item["user_id"]
    if receiver_id == user["user_id"]:
        raise HTTPException(400, "You cannot message yourself")

    message_id = f"mmsg_{uuid.uuid4().hex[:12]}"
    now = datetime.now(timezone.utc).isoformat()
    sb.table("marketplace_messages").insert({
        "message_id": message_id,
        "item_id": item_id,
        "sender_id": user["user_id"],
        "receiver_id": receiver_id,
        "content": payload.content.strip(),
        "created_at": now
    }).execute()
    return {"ok": True, "message_id": message_id}

@api_router.get("/marketplace/items/{item_id}/messages")
def get_market_messages(item_id: str, user: dict = Depends(get_current_user)):
    messages = sb.table("marketplace_messages")\
        .select("*")\
        .eq("item_id", item_id)\
        .or_(f"sender_id.eq.{user['user_id']},receiver_id.eq.{user['user_id']}")\
        .order("created_at", desc=False)\
        .execute().data or []

    # Enrich with sender's display name and profile picture
    sender_ids = list({m["sender_id"] for m in messages})
    profiles = sb.table("user_profiles").select("user_id, display_name, profile_image").in_("user_id", sender_ids).execute().data or []
    pmap = {p["user_id"]: p for p in profiles}
    for m in messages:
        p = pmap.get(m["sender_id"], {})
        m["sender_name"] = p.get("display_name") or "Unknown"
        m["sender_picture"] = p.get("profile_image") or ""
    return messages

# ---------- Mount router ----------
app.include_router(api_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
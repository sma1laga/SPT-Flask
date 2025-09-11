import os, hashlib, json, re, datetime as dt
from flask import Flask, request, Response, jsonify
from sqlalchemy import create_engine, text
import geoip2.database
from user_agents import parse as ua_parse

DATABASE_URL   = os.getenv("DATABASE_URL")
BASE_DOMAIN    = os.getenv("BASE_DOMAIN", "analytics.local")
ALLOWED_ORIGINS = [o.strip() for o in os.getenv("ALLOWED_ORIGINS","").split(",") if o.strip()]
SECRET_KEY     = os.getenv("SECRET_KEY","dev")
SALT           = os.getenv("SALT","devsalt")
ADMIN_USER     = os.getenv("ADMIN_USER","admin")
ADMIN_PASS     = os.getenv("ADMIN_PASS","admin")
GEOIP_DB       = os.getenv("GEOIP_DB","/app/geoip/GeoLite2-Country.mmdb")

app = Flask(__name__)
app.secret_key = SECRET_KEY
engine = create_engine(DATABASE_URL, pool_pre_ping=True)

with engine.begin() as c:
    c.execute(text("""
    CREATE TABLE IF NOT EXISTS events (
      id BIGSERIAL PRIMARY KEY,
      ts TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      site_id TEXT NOT NULL,
      session_id TEXT,
      user_hash TEXT,
      country TEXT,
      path TEXT,
      referrer TEXT,
      ua TEXT,
      is_bot BOOLEAN DEFAULT FALSE,
      tz TEXT,
      lang TEXT,
      screen_w INT,
      screen_h INT,
      event_name TEXT DEFAULT 'pageview',
      event_data JSONB
    );
    CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts);
    CREATE INDEX IF NOT EXISTS idx_events_path ON events(path);
    CREATE INDEX IF NOT EXISTS idx_events_site ON events(site_id);
    """))

geo_reader = geoip2.database.Reader(GEOIP_DB)
BOT_RE = re.compile(r"(bot|crawler|spider|preview|facebookexternalhit|skypeuripreview|slurp|bingpreview)", re.I)

def origin_allowed(origin: str) -> bool:
    if not ALLOWED_ORIGINS: return True
    return origin in ALLOWED_ORIGINS

def client_ip():
    fwd = request.headers.get("X-Forwarded-For", "")
    return fwd.split(",")[0].strip() if fwd else (request.remote_addr or "")

def ip_hash(ip: str) -> str:
    # truncate last octet, then daily-salted hash (privacy-friendly uniqueness per day)
    parts = ip.split(".")
    if len(parts) == 4:
        ip = ".".join(parts[:3]) + ".0"
    day = dt.datetime.utcnow().strftime("%Y-%m-%d")
    return hashlib.sha256((SALT + "|" + ip + "|" + day).encode()).hexdigest()[:32]

def country_from_ip(ip: str) -> str:
    try:
        return geo_reader.country(ip).country.iso_code or "ZZ"
    except Exception:
        return "ZZ"

def is_bot(ua_str: str) -> bool:
    if not ua_str: return True
    if BOT_RE.search(ua_str): return True
    return ua_parse(ua_str).is_bot

@app.after_request
def cors(resp):
    origin = request.headers.get("Origin")
    if origin and origin_allowed(origin):
        resp.headers["Access-Control-Allow-Origin"] = origin
        resp.headers["Vary"] = "Origin"
        resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp

@app.route("/track.js")
def track_js():
    # Minimal tracker: sends automatic pageview & exposes window.SPTA for custom events.
    code = '''
(()=>{try{
  if(navigator.doNotTrack=="1"||window.doNotTrack=="1"){return;}
  const s=[...document.getElementsByTagName("script")].find(x=>/track\\.js/.test(x.src));
  const siteId=s?s.getAttribute("data-site-id")||"default":"default";
  const base={site_id:siteId,
    path:location.pathname+location.search,
    referrer:document.referrer||null,
    tz:Intl.DateTimeFormat().resolvedOptions().timeZone||null,
    lang:navigator.language||null,
    sw:screen&&screen.width||null,
    sh:screen&&screen.height||null,
    title:document.title||null};
  const send=(obj)=>{ const b=new Blob([JSON.stringify(obj)],{type:"application/json"}); navigator.sendBeacon("/collect", b); };
  send(base);
  window.SPTA=(name,data)=>{ try{ send(Object.assign({},base,{event_name:name,event_data:data||{}})); }catch(_){} };
  document.addEventListener("visibilitychange",()=>{ if(document.visibilityState==="hidden"){ send(base); }});
}catch(_){}})();
'''
    return Response(code, mimetype="application/javascript")

@app.route("/collect", methods=["POST","OPTIONS"])
def collect():
    if request.method == "OPTIONS": return ("",204)
    if request.headers.get("DNT")=="1": return ("",204)

    origin = request.headers.get("Origin")
    if origin and not origin_allowed(origin):
        return ("",204)

    payload = request.get_json(force=True, silent=True) or {}
    ua_str  = request.headers.get("User-Agent","")
    if is_bot(ua_str): return ("",204)

    site_id    = (payload.get("site_id") or "default")[:64]
    path       = (payload.get("path") or "/")[:2048]
    referrer   = payload.get("referrer")
    tz         = payload.get("tz")
    lang       = payload.get("lang")
    sw         = payload.get("sw")
    sh         = payload.get("sh")
    event_name = (payload.get("event_name") or "pageview")[:64]
    event_data = payload.get("event_data") or {}

    ip = client_ip()
    userhash = ip_hash(ip)                     # daily salted truncated IP
    country  = country_from_ip(ip)
    day = dt.datetime.utcnow().strftime("%Y-%m-%d")
    ua_key = ua_str[:80] if ua_str else ""
    session = hashlib.sha256((userhash+"|"+ua_key+"|"+day).encode()).hexdigest()[:32]

    with engine.begin() as c:
        c.execute(text("""
          INSERT INTO events(
            site_id, session_id, user_hash, country, path, referrer, ua, tz, lang,
            screen_w, screen_h, event_name, event_data, is_bot
          ) VALUES(
            :site_id,:session_id,:user_hash,:country,:path,:referrer,:ua,:tz,:lang,
            :sw,:sh,:event_name,CAST(:event_data AS JSONB),FALSE
          )
        """), dict(site_id=site_id, session_id=session, user_hash=userhash, country=country,
                   path=path, referrer=referrer, ua=ua_str[:255], tz=tz, lang=lang,
                   sw=sw, sh=sh, event_name=event_name, event_data=json.dumps(event_data)))
    return ("",204)

from functools import wraps
def require_basic_auth(f):
    @wraps(f)
    def wrapper(*a, **kw):
        auth = request.authorization
        if not auth or auth.username!=ADMIN_USER or auth.password!=ADMIN_PASS:
            return Response("Auth required",401,{"WWW-Authenticate":"Basic realm=\"Analytics\""})
        return f(*a, **kw)
    return wrapper

@app.route("/dashboard")
@require_basic_auth
def dashboard():
    with engine.begin() as c:
        row = c.execute(text("""
          WITH base AS (
            SELECT date_trunc('day', ts) d, path, event_name, session_id, country
            FROM events
            WHERE ts >= NOW() - INTERVAL '30 days'
          )
          SELECT
            (SELECT json_agg(x) FROM (
               SELECT d::date AS day, COUNT(*) AS pageviews,
                      COUNT(DISTINCT session_id) AS sessions
               FROM base WHERE event_name='pageview'
               GROUP BY d ORDER BY d
            ) x) AS timeseries,
            (SELECT json_agg(x) FROM (
               SELECT path, COUNT(*) AS cnt FROM base WHERE event_name='pageview'
               GROUP BY path ORDER BY cnt DESC LIMIT 30
            ) x) AS top_pages,
            (SELECT json_agg(x) FROM (
               SELECT country, COUNT(*) AS cnt FROM base WHERE event_name='pageview'
               GROUP BY country ORDER BY cnt DESC
            ) x) AS countries,
            (SELECT json_agg(x) FROM (
               SELECT event_name, COUNT(*) AS cnt FROM base
               GROUP BY event_name ORDER BY cnt DESC
            ) x) AS events;
        """)).mappings().first()
    return jsonify(dict(row or {}))

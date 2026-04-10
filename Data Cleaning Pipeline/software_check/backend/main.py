from __future__ import annotations

import json
import os
import secrets
import tempfile
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from fastapi import FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field


LABELS = ["Irrelevant", "Neutral", "Positive", "Negative"]
SESSION_TTL_SECONDS = 60 * 60 * 24

ROOT = Path(__file__).resolve().parents[1]
PROFILES_PATH = ROOT / "profiles.json"

DEFAULT_ALLOWED_ORIGINS = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]


class LoginRequest(BaseModel):
    password: str


class SelectProfileRequest(BaseModel):
    profile_id: str


class ReviewInput(BaseModel):
    decision: Literal["accept", "reject"]
    final_label: str | None = None
    notes: str | None = None


class CommentReviewInput(ReviewInput):
    comment_id: str


class SaveReviewRequest(BaseModel):
    post_review: ReviewInput
    comment_reviews: list[CommentReviewInput] = Field(default_factory=list)


ReviewInput.model_rebuild()
CommentReviewInput.model_rebuild()
SaveReviewRequest.model_rebuild()


class SessionStore:
    def __init__(self) -> None:
        self._sessions: dict[str, dict[str, Any]] = {}
        self._lock = threading.Lock()

    def create(self) -> str:
        token = secrets.token_urlsafe(32)
        with self._lock:
            self._sessions[token] = {
                "created_at": now_iso(),
                "last_seen_at": now_iso(),
                "profile_id": None,
            }
        return token

    def get(self, token: str) -> dict[str, Any]:
        with self._lock:
            session = self._sessions.get(token)
            if session is None:
                raise HTTPException(status_code=401, detail="Invalid session.")

            age = seconds_since(session["last_seen_at"])
            if age > SESSION_TTL_SECONDS:
                del self._sessions[token]
                raise HTTPException(status_code=401, detail="Session expired.")

            session["last_seen_at"] = now_iso()
            return dict(session)

    def set_profile(self, token: str, profile_id: str) -> None:
        with self._lock:
            session = self._sessions.get(token)
            if session is None:
                raise HTTPException(status_code=401, detail="Invalid session.")
            session["profile_id"] = profile_id
            session["last_seen_at"] = now_iso()

    def delete(self, token: str) -> None:
        with self._lock:
            self._sessions.pop(token, None)


session_store = SessionStore()
profile_file_locks: dict[str, threading.Lock] = {}
profile_file_lock_guard = threading.Lock()


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def seconds_since(iso_timestamp: str) -> float:
    return (datetime.now(timezone.utc) - datetime.fromisoformat(iso_timestamp)).total_seconds()


def load_json(path: Path, default: Any) -> Any:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return default


def save_json_atomic(path: Path, payload: Any) -> None:
    directory = path.parent
    directory.mkdir(parents=True, exist_ok=True)
    fd, temp_path = tempfile.mkstemp(prefix=".tmp_", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
        os.replace(temp_path, path)
    except Exception:
        try:
            os.unlink(temp_path)
        except FileNotFoundError:
            pass
        raise


def get_password() -> str:
    return os.environ.get("REVIEW_APP_PASSWORD", "password123")


def get_allowed_origins() -> list[str]:
    raw = os.environ.get("REVIEW_ALLOWED_ORIGINS")
    if not raw:
        return DEFAULT_ALLOWED_ORIGINS
    return [origin.strip() for origin in raw.split(",") if origin.strip()]


def require_token(authorization: str | None) -> str:
    if not authorization:
        raise HTTPException(status_code=401, detail="Missing Authorization header.")
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        raise HTTPException(status_code=401, detail="Invalid Authorization header.")
    return authorization[len(prefix):].strip()


def load_profiles() -> list[dict[str, Any]]:
    payload = load_json(PROFILES_PATH, {"profiles": []})
    profiles = payload.get("profiles")
    if not isinstance(profiles, list):
        raise HTTPException(status_code=500, detail="profiles.json is invalid.")
    return profiles


def get_profile(profile_id: str) -> dict[str, Any]:
    for profile in load_profiles():
        if profile.get("id") == profile_id:
            return profile
    raise HTTPException(status_code=404, detail="Unknown profile.")


def get_profile_db_path(profile_id: str) -> Path:
    profile = get_profile(profile_id)
    db_file = profile.get("db_file")
    if not isinstance(db_file, str) or not db_file:
        raise HTTPException(status_code=500, detail="Profile db_file is invalid.")
    return ROOT / db_file


def get_profile_file_lock(profile_id: str) -> threading.Lock:
    with profile_file_lock_guard:
        if profile_id not in profile_file_locks:
            profile_file_locks[profile_id] = threading.Lock()
        return profile_file_locks[profile_id]


def load_profile_db(profile_id: str) -> dict[str, Any]:
    payload = load_json(get_profile_db_path(profile_id), {})
    if not isinstance(payload, dict):
        raise HTTPException(status_code=500, detail="Profile database is invalid.")
    return payload


def save_profile_db(profile_id: str, payload: dict[str, Any]) -> None:
    save_json_atomic(get_profile_db_path(profile_id), payload)


def get_selected_profile(token: str) -> dict[str, Any]:
    session = session_store.get(token)
    profile_id = session.get("profile_id")
    if not isinstance(profile_id, str) or not profile_id:
        raise HTTPException(status_code=400, detail="No profile selected.")
    return get_profile(profile_id)


def normalize_review(
    review_input: ReviewInput | CommentReviewInput,
    llm_label: Any,
    reviewer_id: str,
) -> dict[str, Any]:
    if review_input.decision == "accept":
        if llm_label not in LABELS:
            raise HTTPException(status_code=400, detail="Cannot accept an item without an LLM label.")
        final_label = llm_label
    else:
        if review_input.final_label not in LABELS:
            raise HTTPException(status_code=400, detail="Rejected items must choose a valid alternative label.")
        final_label = review_input.final_label

    return {
        "status": "reviewed",
        "decision": review_input.decision,
        "final_label": final_label,
        "reviewed_by": reviewer_id,
        "reviewed_at": now_iso(),
        "notes": review_input.notes.strip() if isinstance(review_input.notes, str) and review_input.notes.strip() else None,
    }


def is_comment_complete(comment: dict[str, Any]) -> bool:
    review = comment.get("review") or {}
    return review.get("status") == "reviewed" and review.get("decision") in {"accept", "reject"}


def is_item_complete(item: dict[str, Any]) -> bool:
    review = item.get("review") or {}
    post_complete = review.get("status") == "reviewed" and review.get("decision") in {"accept", "reject"}
    if not post_complete:
        return False
    comments = item.get("comments")
    if not isinstance(comments, list):
        return True
    return all(is_comment_complete(comment) for comment in comments if isinstance(comment, dict))


def compute_progress(db_payload: dict[str, Any]) -> dict[str, Any]:
    items = db_payload.get("items")
    if not isinstance(items, list):
        items = []
    total = len(items)
    reviewed = sum(1 for item in items if isinstance(item, dict) and is_item_complete(item))
    return {
        "total": total,
        "reviewed": reviewed,
        "remaining": max(total - reviewed, 0),
        "percent": round((reviewed / total) * 100, 2) if total else 0.0,
    }


def augment_item(item: dict[str, Any]) -> dict[str, Any]:
    payload = dict(item)
    comments = payload.get("comments")
    if not isinstance(comments, list):
        comments = []
    comment_rows = [comment for comment in comments if isinstance(comment, dict)]
    payload["computed"] = {
        "is_complete": is_item_complete(item),
        "comment_count": len(comment_rows),
        "reviewed_comment_count": sum(1 for comment in comment_rows if is_comment_complete(comment)),
    }
    return payload


def item_preview(item: dict[str, Any], limit: int = 180) -> str | None:
    post = item.get("post") or {}
    for candidate in [post.get("title"), post.get("text")]:
        if candidate is None:
            continue
        text = str(candidate).strip()
        if not text:
            continue
        if len(text) <= limit:
            return text
        return text[:limit].rstrip() + "..."
    return None


def history_entry(item: dict[str, Any]) -> dict[str, Any]:
    comments = [
        comment
        for comment in item.get("comments", [])
        if isinstance(comment, dict)
    ]
    return {
        "item_id": item.get("item_id"),
        "position": item.get("position"),
        "source": (item.get("post") or {}).get("source"),
        "type": (item.get("post") or {}).get("type"),
        "author": (item.get("post") or {}).get("author"),
        "title": (item.get("post") or {}).get("title"),
        "date": (item.get("post") or {}).get("date"),
        "preview": item_preview(item),
        "llm_label": (item.get("llm") or {}).get("label"),
        "decision": (item.get("review") or {}).get("decision"),
        "final_label": (item.get("review") or {}).get("final_label"),
        "reviewed_at": (item.get("review") or {}).get("reviewed_at"),
        "comment_count": len(comments),
        "reviewed_comment_count": sum(1 for comment in comments if is_comment_complete(comment)),
        "accepted_comment_count": sum(
            1 for comment in comments if (comment.get("review") or {}).get("decision") == "accept"
        ),
        "rejected_comment_count": sum(
            1 for comment in comments if (comment.get("review") or {}).get("decision") == "reject"
        ),
    }


def reviewed_history(db_payload: dict[str, Any], limit: int | None = None) -> list[dict[str, Any]]:
    rows = [
        history_entry(item)
        for item in list_items(db_payload)
        if is_item_complete(item)
    ]
    rows.sort(key=lambda row: row.get("reviewed_at") or "", reverse=True)
    if isinstance(limit, int) and limit >= 0:
        return rows[:limit]
    return rows


def list_items(db_payload: dict[str, Any]) -> list[dict[str, Any]]:
    items = db_payload.get("items")
    if not isinstance(items, list):
        return []
    return [item for item in items if isinstance(item, dict)]


def find_item(db_payload: dict[str, Any], item_id: str) -> dict[str, Any]:
    for item in list_items(db_payload):
        if item.get("item_id") == item_id:
            return item
    raise HTTPException(status_code=404, detail="Item not found.")


def next_incomplete_item(db_payload: dict[str, Any]) -> dict[str, Any] | None:
    for item in list_items(db_payload):
        if not is_item_complete(item):
            return item
    return None


app = FastAPI(title="Software Check Review API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_allowed_origins(),
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/auth/login")
def login(body: LoginRequest) -> dict[str, Any]:
    if body.password != get_password():
        raise HTTPException(status_code=401, detail="Invalid password.")

    token = session_store.create()
    return {
        "token": token,
        "profiles": load_profiles(),
    }


@app.post("/api/auth/logout")
def logout(authorization: str | None = Header(default=None)) -> dict[str, bool]:
    token = require_token(authorization)
    session_store.delete(token)
    return {"ok": True}


@app.post("/api/profile/select")
def select_profile(
    body: SelectProfileRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    token = require_token(authorization)
    session_store.get(token)
    profile = get_profile(body.profile_id)
    session_store.set_profile(token, body.profile_id)

    db_payload = load_profile_db(body.profile_id)
    return {
        "profile": profile,
        "progress": compute_progress(db_payload),
    }


@app.get("/api/profile")
def current_profile(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    token = require_token(authorization)
    session = session_store.get(token)
    selected_profile = None
    progress = None
    profile_id = session.get("profile_id")
    if isinstance(profile_id, str) and profile_id:
        selected_profile = get_profile(profile_id)
        progress = compute_progress(load_profile_db(profile_id))

    return {
        "profiles": load_profiles(),
        "selected_profile": selected_profile,
        "progress": progress,
    }


@app.get("/api/progress")
def progress(authorization: str | None = Header(default=None)) -> dict[str, Any]:
    token = require_token(authorization)
    profile = get_selected_profile(token)
    db_payload = load_profile_db(profile["id"])
    return {
        "profile": profile,
        "progress": compute_progress(db_payload),
    }


@app.get("/api/history")
def history(
    authorization: str | None = Header(default=None),
    limit: int | None = Query(default=None, ge=0),
) -> dict[str, Any]:
    token = require_token(authorization)
    profile = get_selected_profile(token)
    db_payload = load_profile_db(profile["id"])
    return {
        "profile": profile,
        "progress": compute_progress(db_payload),
        "items": reviewed_history(db_payload, limit),
    }


@app.get("/api/items/current")
def current_item(
    authorization: str | None = Header(default=None),
    item_id: str | None = Query(default=None),
) -> dict[str, Any]:
    token = require_token(authorization)
    profile = get_selected_profile(token)
    db_payload = load_profile_db(profile["id"])

    item = find_item(db_payload, item_id) if item_id else next_incomplete_item(db_payload)
    if item is None:
        return {
            "profile": profile,
            "progress": compute_progress(db_payload),
            "item": None,
        }

    return {
        "profile": profile,
        "progress": compute_progress(db_payload),
        "item": augment_item(item),
    }


@app.get("/api/items/{item_id}")
def get_item(item_id: str, authorization: str | None = Header(default=None)) -> dict[str, Any]:
    token = require_token(authorization)
    profile = get_selected_profile(token)
    db_payload = load_profile_db(profile["id"])
    item = find_item(db_payload, item_id)

    return {
        "profile": profile,
        "progress": compute_progress(db_payload),
        "item": augment_item(item),
    }


@app.post("/api/items/{item_id}/review")
def save_review(
    item_id: str,
    body: SaveReviewRequest,
    authorization: str | None = Header(default=None),
) -> dict[str, Any]:
    token = require_token(authorization)
    profile = get_selected_profile(token)
    profile_id = profile["id"]

    with get_profile_file_lock(profile_id):
        db_payload = load_profile_db(profile_id)
        item = find_item(db_payload, item_id)
        item["review"] = normalize_review(body.post_review, item.get("llm", {}).get("label"), profile_id)

        comment_index = {
            comment.get("comment_id"): comment
            for comment in item.get("comments", [])
            if isinstance(comment, dict) and isinstance(comment.get("comment_id"), str)
        }

        for comment_review in body.comment_reviews:
            comment = comment_index.get(comment_review.comment_id)
            if comment is None:
                raise HTTPException(status_code=400, detail=f"Comment {comment_review.comment_id} does not exist on this item.")

            comment["review"] = normalize_review(comment_review, comment.get("llm", {}).get("label"), profile_id)

        incomplete_comments = [
            comment.get("comment_id")
            for comment in item.get("comments", [])
            if isinstance(comment, dict)
            if not is_comment_complete(comment)
        ]
        if incomplete_comments:
            raise HTTPException(
                status_code=400,
                detail=f"All comments must be reviewed before saving. Missing: {', '.join(incomplete_comments)}",
            )

        save_profile_db(profile_id, db_payload)

    updated_db = load_profile_db(profile_id)
    next_item = next_incomplete_item(updated_db)
    return {
        "ok": True,
        "profile": profile,
        "progress": compute_progress(updated_db),
        "saved_item_id": item_id,
        "next_item_id": next_item.get("item_id") if next_item else None,
    }

import os
import json
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Labeling API")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = Path(__file__).resolve().parent
SECRET_KEY = os.environ.get("LABELING_SECRET_KEY", "secret123")
LABELERS = ["ananya", "bryan", "leonard", "ryan"]


def get_db_path(labeler_name: str) -> Path:
    if labeler_name not in LABELERS:
        raise HTTPException(status_code=404, detail=f"Labeler {labeler_name} not found")
    return BASE_DIR / f"db_{labeler_name}.json"


def read_db(labeler_name: str) -> list:
    db_path = get_db_path(labeler_name)
    if not db_path.exists():
        return []
    try:
        with open(db_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []


def write_db(labeler_name: str, data: list):
    db_path = get_db_path(labeler_name)
    with open(db_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


# --- 1. Authentication / Secret Key ---

class VerifyRequest(BaseModel):
    secret_key: str

@app.post("/api/auth/verify")
def verify_secret(req: VerifyRequest):
    if req.secret_key == SECRET_KEY:
        return {"valid": True}
    return Response(
        content=json.dumps({"valid": False, "message": "Invalid secret key"}),
        status_code=401,
        media_type="application/json"
    )


# --- 2. Labeler Operations ---

@app.get("/api/labelers")
def get_labelers():
    return LABELERS

@app.get("/api/labelers/{labeler_name}/items/next")
def get_next_item(labeler_name: str):
    data = read_db(labeler_name)
    for item in data:
        # Check if item is labeled
        if not item.get("is_labeled", False):
            # Extract fields that are typically requested, with fallback to item contents
            return {
                "ID": item.get("ID"),
                "Source": item.get("Source", ""),
                "Type": item.get("Type", ""),
                "Author": item.get("Author", ""),
                "Title": item.get("Title", ""),
                "Text": item.get("Text", ""),
                "Date": item.get("Date", ""),
                "Comments": item.get("Comments") or [] # Included in case text is within comments
            }
    
    # If we iterate through all items and find no unlabeled items
    return Response(
        content=json.dumps({"message": "No more items left to label."}),
        status_code=404,
        media_type="application/json"
    )

@app.get("/api/labelers/{labeler_name}/stats")
def get_stats(labeler_name: str):
    data = read_db(labeler_name)
    total_assigned = len(data)
    labeled_count = sum(1 for item in data if item.get("is_labeled", False))
    remaining_count = total_assigned - labeled_count
    return {
        "labeled_count": labeled_count,
        "remaining_count": remaining_count,
        "total_assigned": total_assigned
    }

@app.get("/api/labelers/{labeler_name}/items")
def get_items(labeler_name: str):
    data = read_db(labeler_name)
    result = []
    for item in data:
        result.append({
            "ID": item.get("ID"),
            "Title": item.get("Title", ""),
            "Text": item.get("Text", ""),
            "is_labeled": item.get("is_labeled", False),
            "label": item.get("label", None)
        })
    return result


# --- 3. Label Recording ---

class LabelRequest(BaseModel):
    labeler_name: str
    item_id: str
    label: str
    comment_labels: dict[str, str] = {}

@app.post("/api/labels", status_code=201)
def record_label(req: LabelRequest):
    data = read_db(req.labeler_name)
    found = False
    
    for item in data:
        if str(item.get("ID")) == str(req.item_id):
            # Validate comment labels
            comments = item.get("Comments") or []
            for c in comments:
                cid = str(c.get("comment_id"))
                if cid not in req.comment_labels:
                    raise HTTPException(status_code=400, detail=f"Missing label for comment {cid}")
            
            # Validations passed, apply labels
            item["is_labeled"] = True
            item["label"] = req.label
            
            for c in comments:
                cid = str(c.get("comment_id"))
                c["is_labeled"] = True
                c["label"] = req.comment_labels[cid]
                
            found = True
            break
            
    if not found:
        raise HTTPException(status_code=404, detail="Item not found")
        
    write_db(req.labeler_name, data)
    return {"success": True, "message": "Label recorded successfully."}


# --- 4. Generic CRUD for Items (Database Management) ---

@app.get("/api/items")
def get_all_items(page: int = 1, limit: int = 50):
    all_items = []
    for l in LABELERS:
        try:
            data = read_db(l)
            for d in data:
                d["_labeler"] = l
                all_items.append(d)
        except HTTPException:
            pass
            
    start = (page - 1) * limit
    end = start + limit
    
    return {
        "page": page,
        "limit": limit,
        "total": len(all_items),
        "items": all_items[start:end]
    }

@app.get("/api/items/{id}")
def get_item(id: str):
    for l in LABELERS:
        data = read_db(l)
        for item in data:
            if str(item.get("ID")) == id:
                item["_labeler"] = l
                return item
    raise HTTPException(status_code=404, detail="Item not found")

@app.post("/api/items", status_code=201)
async def add_item(request: Request):
    try:
        new_item = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
        
    labeler_name = new_item.get("_labeler", LABELERS[0])
    if "_labeler" in new_item:
        del new_item["_labeler"]
        
    data = read_db(labeler_name)
    data.append(new_item)
    write_db(labeler_name, data)
    
    return {"success": True, "message": "Item added successfully.", "id": new_item.get("ID")}

@app.put("/api/items/{id}")
async def update_item(id: str, request: Request):
    try:
        update_data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")
        
    for l in LABELERS:
        data = read_db(l)
        for item in data:
            if str(item.get("ID")) == id:
                item.update(update_data)
                write_db(l, data)
                return {"success": True, "item": item}
                
    raise HTTPException(status_code=404, detail="Item not found")

@app.delete("/api/items/{id}")
def delete_item(id: str):
    deleted_any = False
    
    for l in LABELERS:
        data = read_db(l)
        original_length = len(data)
        data = [item for item in data if str(item.get("ID")) != id]
        
        if len(data) < original_length:
            write_db(l, data)
            deleted_any = True
            
    if deleted_any:
        return {"success": True, "message": "Item deleted."}
        
    raise HTTPException(status_code=404, detail="Item not found")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

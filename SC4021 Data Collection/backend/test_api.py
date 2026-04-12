import pytest
import json
import os
from pathlib import Path
from fastapi.testclient import TestClient
from typing import Generator

# Import the app and configuration variables from api.py
from api import app, LABELERS, BASE_DIR

client = TestClient(app)

# --- Test Data & Fixtures ---

MOCK_DATA = [
    {
        "ID": "item_1",
        "Source": "Test Source",
        "Type": "Article",
        "Title": "Test Title 1",
        "Text": "This is test text 1.",
        "is_labeled": False,
        "label": None,
        "Comments": [
            {
                "comment_id": "c1",
                "Text": "A comment"
            }
        ]
    },
    {
        "ID": "item_2",
        "Title": "Test Title 2",
        "Text": "This is test text 2.",
        "is_labeled": True,
        "label": "positive"
    }
]

@pytest.fixture(autouse=True)
def setup_teardown_dbs() -> Generator:
    """
    Sets up temporary mock databases before each test and tears them down after.
    Uses the exact same file paths as api.py but temporarily clobbers/restores them 
    for the specific labelers under test, or just cleans them up.
    """
    # Backup original data if it exists
    backups = {}
    for labeler in LABELERS:
        db_path = BASE_DIR / f"db_{labeler}.json"
        if db_path.exists():
            with open(db_path, "r", encoding="utf-8") as f:
                backups[labeler] = f.read()
        
        # Write fresh mock data for testing
        with open(db_path, "w", encoding="utf-8") as f:
            json.dump(MOCK_DATA, f)

    yield  # Run the test

    # Restore original data
    for labeler in LABELERS:
        db_path = BASE_DIR / f"db_{labeler}.json"
        if labeler in backups:
            with open(db_path, "w", encoding="utf-8") as f:
                f.write(backups[labeler])
        else:
            if db_path.exists():
                os.remove(db_path)

# --- 1. Authentication Tests ---

def test_verify_secret_valid():
    # Uses the default "secret123" from api.py
    response = client.post("/api/auth/verify", json={"secret_key": "secret123"})
    assert response.status_code == 200
    assert response.json() == {"valid": True}

def test_verify_secret_invalid():
    response = client.post("/api/auth/verify", json={"secret_key": "wrongsecret"})
    assert response.status_code == 401
    assert response.json() == {"valid": False, "message": "Invalid secret key"}

# --- 2. Labeler Operations Tests ---

def test_get_labelers():
    response = client.get("/api/labelers")
    assert response.status_code == 200
    assert response.json() == LABELERS

def test_get_next_item_success():
    labeler = LABELERS[0]
    response = client.get(f"/api/labelers/{labeler}/items/next")
    assert response.status_code == 200
    data = response.json()
    assert data["ID"] == "item_1"
    assert "Text" in data

def test_get_next_item_null_comments():
    labeler = LABELERS[0]
    db_path = BASE_DIR / f"db_{labeler}.json"
    with open(db_path, "w", encoding="utf-8") as f:
        json.dump([
            {
                "ID": "item_null_comments",
                "is_labeled": False,
                "Comments": None
            }
        ], f)
    
    response = client.get(f"/api/labelers/{labeler}/items/next")
    assert response.status_code == 200
    assert response.json()["Comments"] == []

def test_get_next_item_exhausted():
    labeler = LABELERS[0]
    # Mark the only unlabeled item as labeled
    db_path = BASE_DIR / f"db_{labeler}.json"
    with open(db_path, "w", encoding="utf-8") as f:
        json.dump([
            {**MOCK_DATA[0], "is_labeled": True},
            MOCK_DATA[1]
        ], f)
        
    response = client.get(f"/api/labelers/{labeler}/items/next")
    assert response.status_code == 404
    assert response.json() == {"message": "No more items left to label."}

def test_get_stats():
    labeler = LABELERS[0]
    response = client.get(f"/api/labelers/{labeler}/stats")
    assert response.status_code == 200
    stats = response.json()
    assert stats["total_assigned"] == 2
    assert stats["labeled_count"] == 1
    assert stats["remaining_count"] == 1

def test_get_items():
    labeler = LABELERS[0]
    response = client.get(f"/api/labelers/{labeler}/items")
    assert response.status_code == 200
    items = response.json()
    assert len(items) == 2
    # Ensure they return the required structure
    assert "is_labeled" in items[0]
    assert "label" in items[0]

def test_invalid_labeler_404():
    response = client.get("/api/labelers/invalid_user/items")
    assert response.status_code == 404

# --- 3. Label Recording Tests ---

def test_record_label_success():
    labeler = LABELERS[0]
    payload = {
        "labeler_name": labeler,
        "item_id": "item_1",
        "label": "negative",
        "comment_labels": {
            "c1": "positive"
        }
    }
    response = client.post("/api/labels", json=payload)
    assert response.status_code == 201
    assert response.json() == {"success": True, "message": "Label recorded successfully."}
    
    # Verify state change
    db_path = BASE_DIR / f"db_{labeler}.json"
    with open(db_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data[0]["is_labeled"] == True
        assert data[0]["label"] == "negative"
        assert data[0]["Comments"][0]["is_labeled"] == True
        assert data[0]["Comments"][0]["label"] == "positive"

def test_record_label_missing_comment():
    labeler = LABELERS[0]
    payload = {
        "labeler_name": labeler,
        "item_id": "item_1",
        "label": "positive",
        "comment_labels": {}
    }
    response = client.post("/api/labels", json=payload)
    assert response.status_code == 400
    assert "Missing label for comment c1" in response.json()["detail"]

def test_record_label_not_found():
    payload = {
        "labeler_name": LABELERS[0],
        "item_id": "non_existent",
        "label": "positive"
    }
    response = client.post("/api/labels", json=payload)
    assert response.status_code == 404

def test_record_label_null_comments():
    labeler = LABELERS[0]
    db_path = BASE_DIR / f"db_{labeler}.json"
    with open(db_path, "w", encoding="utf-8") as f:
        json.dump([
            {
                "ID": "item_null_comments",
                "is_labeled": False,
                "Comments": None
            }
        ], f)
    
    payload = {
        "labeler_name": labeler,
        "item_id": "item_null_comments",
        "label": "positive",
        "comment_labels": {}
    }
    response = client.post("/api/labels", json=payload)
    assert response.status_code == 201
    assert response.json()["success"] == True

# --- 4. CRUD Item Tests ---

def test_get_all_items():
    response = client.get("/api/items")
    assert response.status_code == 200
    data = response.json()
    assert "items" in data
    # 2 items per labeler * len(LABELERS)
    assert len(data["items"]) == 2 * len(LABELERS)

def test_get_item_by_id():
    response = client.get("/api/items/item_2")
    assert response.status_code == 200
    data = response.json()
    assert data["ID"] == "item_2"

def test_add_item():
    new_item = {
        "ID": "item_3",
        "Title": "New item",
        "_labeler": LABELERS[1]
    }
    response = client.post("/api/items", json=new_item)
    assert response.status_code == 201
    
    # Verify it was added to the correct db
    db_path = BASE_DIR / f"db_{LABELERS[1]}.json"
    with open(db_path, "r") as f:
        data = json.load(f)
        assert len(data) == 3
        assert data[2]["ID"] == "item_3"

def test_update_item():
    update_data = {"Title": "Updated Title"}
    response = client.put("/api/items/item_1", json=update_data)
    assert response.status_code == 200
    
    # Verify the update happened
    verify_response = client.get("/api/items/item_1")
    assert verify_response.json()["Title"] == "Updated Title"

def test_delete_item():
    response = client.delete("/api/items/item_2")
    assert response.status_code == 200
    
    # Verify it is gone
    verify_response = client.get("/api/items/item_2")
    assert verify_response.status_code == 404

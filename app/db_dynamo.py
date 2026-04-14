"""
DynamoDB backend for WikiDelve.

Drop-in replacement for SQLite when DB_BACKEND=dynamodb is set.
Uses a single-table design with composite keys for all entity types.

Table: wikidelve (or DB_DYNAMO_TABLE env var)
  PK (partition key) — entity type prefix + id, e.g. "JOB#42", "SRC#42#1"
  SK (sort key) — secondary dimension, e.g. "META", "ROUND#1", "HALL#infra"

Requires: boto3 (already a dependency for S3/Bedrock)
"""

import os
import logging
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from typing import Any

import boto3
from boto3.dynamodb.conditions import Key, Attr

from app.config import (
    COOLDOWN_DAYS,
    AWS_ACCESS_KEY_ID,
    AWS_SECRET_ACCESS_KEY,
    AWS_SESSION_TOKEN,
    BEDROCK_REGION,
)

logger = logging.getLogger("kb-service.dynamo")

TABLE_NAME = os.getenv("DB_DYNAMO_TABLE", "wikidelve")
_table = None

# Auto-incrementing ID counters — stored in DynamoDB as COUNTER#<entity>
_COUNTER_ENTITIES = ("job", "source", "update", "entity", "edge", "room")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_table():
    """Lazy-init DynamoDB table resource."""
    global _table
    if _table is not None:
        return _table

    kwargs: dict = {"region_name": BEDROCK_REGION}
    if AWS_ACCESS_KEY_ID and AWS_SECRET_ACCESS_KEY:
        kwargs["aws_access_key_id"] = AWS_ACCESS_KEY_ID
        kwargs["aws_secret_access_key"] = AWS_SECRET_ACCESS_KEY
        if AWS_SESSION_TOKEN:
            kwargs["aws_session_token"] = AWS_SESSION_TOKEN

    endpoint = os.getenv("DB_DYNAMO_ENDPOINT")
    if endpoint:
        kwargs["endpoint_url"] = endpoint

    dynamodb = boto3.resource("dynamodb", **kwargs)
    _table = dynamodb.Table(TABLE_NAME)
    return _table


def _clean_item(item: dict) -> dict:
    """Convert Decimals back to int/float for JSON compatibility."""
    out = {}
    for k, v in item.items():
        if isinstance(v, Decimal):
            out[k] = int(v) if v == int(v) else float(v)
        else:
            out[k] = v
    return out


def _to_decimal(val: Any) -> Any:
    """Convert floats to Decimal for DynamoDB compatibility."""
    if isinstance(val, float):
        return Decimal(str(val))
    return val


async def _next_id(entity: str) -> int:
    """Atomic auto-increment counter using DynamoDB."""
    table = _get_table()
    resp = table.update_item(
        Key={"PK": f"COUNTER#{entity}", "SK": "COUNTER"},
        UpdateExpression="ADD #val :inc",
        ExpressionAttributeNames={"#val": "value"},
        ExpressionAttributeValues={":inc": 1},
        ReturnValues="UPDATED_NEW",
    )
    return int(resp["Attributes"]["value"])


# ---------------------------------------------------------------------------
# Init
# ---------------------------------------------------------------------------

async def init_db() -> None:
    """Verify the DynamoDB table is reachable. The table is provisioned and
    managed by Terraform (`wikidelve-infra`); we never auto-create it from
    the application — that would diverge from IaC."""
    table = _get_table()
    try:
        _ = table.table_status
        logger.info("DynamoDB backend ready: table=%s", TABLE_NAME)
    except table.meta.client.exceptions.ResourceNotFoundException as exc:
        raise RuntimeError(
            f"DynamoDB table '{TABLE_NAME}' does not exist. "
            f"Run `terraform apply` in wikidelve-infra to provision it."
        ) from exc


# ---------------------------------------------------------------------------
# Research jobs
# ---------------------------------------------------------------------------

async def create_job(
    topic: str,
    *,
    job_type: str = "web",
    source_params: str | None = None,
) -> int:
    job_id = await _next_id("job")
    table = _get_table()
    table.put_item(Item={
        "PK": f"JOB#{job_id}",
        "SK": "META",
        "id": job_id,
        "topic": topic,
        "status": "queued",
        "created_at": _now_iso(),
        "completed_at": None,
        "sources_count": 0,
        "word_count": 0,
        "error": None,
        "added_to_wiki": 0,
        "content": None,
        "job_type": job_type,
        "source_params": source_params,
        # GSI1 for listing by status/time
        "GSI1PK": "JOBS",
        "GSI1SK": f"{_now_iso()}#{job_id}",
    })
    return job_id


async def update_job(job_id: int, **fields) -> None:
    if not fields:
        return
    table = _get_table()
    update_parts = []
    names = {}
    values = {}
    for i, (k, v) in enumerate(fields.items()):
        alias = f"#f{i}"
        val_alias = f":v{i}"
        update_parts.append(f"{alias} = {val_alias}")
        names[alias] = k
        values[val_alias] = _to_decimal(v) if v is not None else None
    table.update_item(
        Key={"PK": f"JOB#{job_id}", "SK": "META"},
        UpdateExpression="SET " + ", ".join(update_parts),
        ExpressionAttributeNames=names,
        ExpressionAttributeValues=values,
    )


async def get_job(job_id: int) -> dict | None:
    table = _get_table()
    resp = table.get_item(Key={"PK": f"JOB#{job_id}", "SK": "META"})
    item = resp.get("Item")
    if not item:
        return None
    return _strip_dynamo_keys(_clean_item(item))


async def delete_job(job_id: int) -> None:
    table = _get_table()
    # Delete sources first
    resp = table.query(KeyConditionExpression=Key("PK").eq(f"SRC#{job_id}"))
    with table.batch_writer() as batch:
        for item in resp.get("Items", []):
            batch.delete_item(Key={"PK": item["PK"], "SK": item["SK"]})
    # Delete job
    table.delete_item(Key={"PK": f"JOB#{job_id}", "SK": "META"})


async def get_jobs(limit: int = 50, compact: bool = False) -> list[dict]:
    """Return recent jobs via GSI1 newest-first.

    DynamoDB pages queries at 1 MB max. With the synthesized ``content``
    field on each job row (~5–10 KB of markdown per completed research
    article), a naïve ``Limit=200`` returns only ~100 jobs before the
    page cap is hit. We now:

      - Pass ``ProjectionExpression`` to strip the heavy fields when the
        caller is fine without them (``compact=True``). Rows shrink
        roughly 50×, so each 1 MB page holds thousands of jobs.
      - Auto-paginate via ``ExclusiveStartKey`` until ``limit`` items
        are collected or the index is exhausted, regardless of mode.
    """
    table = _get_table()

    query_kwargs = {
        "IndexName": "GSI1",
        "KeyConditionExpression": Key("GSI1PK").eq("JOBS"),
        "ScanIndexForward": False,
    }
    if compact:
        # Boto3 will translate these to ExpressionAttributeNames; #pk/#sk
        # avoid the reserved-word clash with the partition/sort keys.
        query_kwargs["ProjectionExpression"] = (
            "#pk, #sk, id, topic, #status, created_at, completed_at, "
            "sources_count, word_count, #err, added_to_wiki, kb, "
            "GSI1PK, GSI1SK"
        )
        query_kwargs["ExpressionAttributeNames"] = {
            "#pk": "PK",
            "#sk": "SK",
            "#status": "status",
            "#err": "error",
        }

    items: list[dict] = []
    remaining = limit
    last_key = None
    while remaining > 0:
        if last_key is not None:
            query_kwargs["ExclusiveStartKey"] = last_key
        query_kwargs["Limit"] = remaining
        resp = table.query(**query_kwargs)
        page = resp.get("Items", [])
        items.extend(page)
        remaining = limit - len(items)
        last_key = resp.get("LastEvaluatedKey")
        if not last_key:
            break
    return [_strip_dynamo_keys(_clean_item(i)) for i in items[:limit]]


async def get_stuck_jobs() -> list[dict]:
    stuck_statuses = {
        "queued", "searching", "searching_round_1", "searching_round_2",
        "searching_round_3", "synthesizing", "reading_pages",
        "downloading_docs", "browser_reading",
    }
    jobs = await get_jobs(limit=200)
    return [j for j in jobs if j.get("status") in stuck_statuses]


async def get_errored_jobs(limit: int = 50) -> list[dict]:
    jobs = await get_jobs(limit=200)
    return [j for j in jobs if j.get("status") == "error"][:limit]


async def reset_job_for_retry(job_id: int):
    await update_job(job_id, status="queued", error=None)


async def get_job_stats() -> dict:
    # Compact mode drops the ``content`` column server-side so rows
    # are tiny and a single DynamoDB page fits thousands of jobs.
    jobs = await get_jobs(limit=10000, compact=True)
    total = len(jobs)
    complete = sum(1 for j in jobs if j.get("status") == "complete")
    errors = sum(1 for j in jobs if j.get("status") == "error")
    cancelled = sum(1 for j in jobs if j.get("status") == "cancelled")
    active = sum(1 for j in jobs if j.get("status") in (
        "queued", "searching", "searching_round_1", "searching_round_2",
        "searching_round_3", "synthesizing", "reading_pages",
        "downloading_docs", "browser_reading",
    ))
    total_words = sum(j.get("word_count", 0) for j in jobs)
    added = sum(1 for j in jobs if j.get("added_to_wiki"))
    return {
        "total": total, "complete": complete, "errors": errors,
        "active": active, "cancelled": cancelled,
        "total_words": total_words, "added_to_wiki": added,
    }


async def check_cooldown(topic: str) -> dict | None:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=COOLDOWN_DAYS)).isoformat()
    jobs = await get_jobs(limit=500, compact=True)
    for j in jobs:
        if (j.get("topic") == topic and j.get("status") == "complete"
                and j.get("created_at", "") > cutoff):
            return j
    return None


# ---------------------------------------------------------------------------
# Research sources
# ---------------------------------------------------------------------------

async def save_sources(job_id: int, sources: list[dict], round_num: int) -> None:
    if not sources:
        return
    table = _get_table()
    with table.batch_writer() as batch:
        for s in sources:
            src_id = await _next_id("source")
            batch.put_item(Item={
                "PK": f"SRC#{job_id}",
                "SK": f"ROUND#{round_num}#{src_id:012d}",
                "id": src_id,
                "job_id": job_id,
                "url": s.get("url", ""),
                "title": s.get("title", ""),
                "content": s.get("content", ""),
                "tier": int(s.get("tier", 3)),
                "round": round_num,
                "selected": 1,
            })


async def get_sources(job_id: int) -> list[dict]:
    table = _get_table()
    resp = table.query(KeyConditionExpression=Key("PK").eq(f"SRC#{job_id}"))
    items = [_strip_dynamo_keys(_clean_item(i)) for i in resp.get("Items", [])]
    items.sort(key=lambda r: (int(r.get("round", 0)), int(r.get("id", 0))))
    return items


async def update_source_selection(job_id: int, source_ids: list[int], selected: bool) -> int:
    if not source_ids:
        return 0
    table = _get_table()
    rows = await get_sources(job_id)
    target_ids = set(int(i) for i in source_ids)
    count = 0
    for r in rows:
        if int(r.get("id", -1)) not in target_ids:
            continue
        round_num = int(r.get("round", 1))
        sk = f"ROUND#{round_num}#{int(r['id']):012d}"
        table.update_item(
            Key={"PK": f"SRC#{job_id}", "SK": sk},
            UpdateExpression="SET selected = :s",
            ExpressionAttributeValues={":s": 1 if selected else 0},
        )
        count += 1
    return count


async def get_selected_sources(job_id: int) -> list[dict]:
    rows = await get_sources(job_id)
    rows = [r for r in rows if int(r.get("selected", 0)) == 1]
    rows.sort(key=lambda r: (int(r.get("tier", 3)), int(r.get("round", 0)), int(r.get("id", 0))))
    return rows


async def select_all_sources(job_id: int, selected: bool = True) -> int:
    rows = await get_sources(job_id)
    count = 0
    table = _get_table()
    for r in rows:
        round_num = int(r.get("round", 1))
        sk = f"ROUND#{round_num}#{int(r['id']):012d}"
        table.update_item(
            Key={"PK": f"SRC#{job_id}", "SK": sk},
            UpdateExpression="SET selected = :s",
            ExpressionAttributeValues={":s": 1 if selected else 0},
        )
        count += 1
    return count


# ---------------------------------------------------------------------------
# Article updates
# ---------------------------------------------------------------------------

async def log_article_update(
    article_slug: str, kb_name: str, job_id: int | None, change_type: str = "created",
) -> None:
    update_id = await _next_id("update")
    table = _get_table()
    table.put_item(Item={
        "PK": f"UPDATE#{update_id}",
        "SK": "META",
        "id": update_id,
        "article_slug": article_slug,
        "kb_name": kb_name,
        "updated_at": _now_iso(),
        "job_id": job_id,
        "change_type": change_type,
        "GSI1PK": "UPDATES",
        "GSI1SK": f"{_now_iso()}#{update_id}",
    })


async def get_article_updates(limit: int = 50) -> list[dict]:
    table = _get_table()
    resp = table.query(
        IndexName="GSI1",
        KeyConditionExpression=Key("GSI1PK").eq("UPDATES"),
        ScanIndexForward=False,
        Limit=limit,
    )
    return [_strip_dynamo_keys(_clean_item(i)) for i in resp.get("Items", [])]


# ---------------------------------------------------------------------------
# Palace classifications
# ---------------------------------------------------------------------------

async def upsert_classification(slug: str, kb: str, hall: str, confidence: float) -> None:
    table = _get_table()
    table.put_item(Item={
        "PK": f"CLASS#{kb}#{slug}",
        "SK": "META",
        "slug": slug,
        "kb": kb,
        "hall": hall,
        "confidence": _to_decimal(confidence),
        "updated_at": _now_iso(),
        "GSI1PK": f"CLASS#{kb}",
        "GSI1SK": f"{hall}#{slug}",
    })


async def get_classifications(kb: str | None = None) -> list[dict]:
    table = _get_table()
    if kb:
        resp = table.query(
            IndexName="GSI1",
            KeyConditionExpression=Key("GSI1PK").eq(f"CLASS#{kb}"),
        )
    else:
        resp = table.scan(FilterExpression=Attr("PK").begins_with("CLASS#") & Attr("SK").eq("META"))
    return [_strip_dynamo_keys(_clean_item(i)) for i in resp.get("Items", [])]


async def get_classifications_by_hall(kb: str, hall: str) -> list[dict]:
    table = _get_table()
    resp = table.query(
        IndexName="GSI1",
        KeyConditionExpression=Key("GSI1PK").eq(f"CLASS#{kb}") & Key("GSI1SK").begins_with(f"{hall}#"),
    )
    items = [_strip_dynamo_keys(_clean_item(i)) for i in resp.get("Items", [])]
    items.sort(key=lambda x: x.get("confidence", 0), reverse=True)
    return items


async def get_article_classification(slug: str, kb: str) -> dict | None:
    table = _get_table()
    resp = table.get_item(Key={"PK": f"CLASS#{kb}#{slug}", "SK": "META"})
    item = resp.get("Item")
    return _strip_dynamo_keys(_clean_item(item)) if item else None


# ---------------------------------------------------------------------------
# Palace rooms
# ---------------------------------------------------------------------------

async def upsert_room(kb: str, name: str, anchor_entity_id: int | None, article_count: int) -> int:
    table = _get_table()
    # Check if room exists
    resp = table.query(
        IndexName="GSI1",
        KeyConditionExpression=Key("GSI1PK").eq(f"ROOMS#{kb}") & Key("GSI1SK").eq(name),
    )
    existing = resp.get("Items", [])
    if existing:
        room_id = int(existing[0]["id"])
        table.update_item(
            Key={"PK": f"ROOM#{room_id}", "SK": "META"},
            UpdateExpression="SET anchor_entity_id = :a, article_count = :c, updated_at = :u",
            ExpressionAttributeValues={
                ":a": anchor_entity_id,
                ":c": article_count,
                ":u": _now_iso(),
            },
        )
        return room_id

    room_id = await _next_id("room")
    table.put_item(Item={
        "PK": f"ROOM#{room_id}",
        "SK": "META",
        "id": room_id,
        "kb": kb,
        "name": name,
        "anchor_entity_id": anchor_entity_id,
        "article_count": article_count,
        "updated_at": _now_iso(),
        "GSI1PK": f"ROOMS#{kb}",
        "GSI1SK": name,
    })
    return room_id


async def add_room_member(room_id: int, slug: str, kb: str, relevance: float = 1.0) -> None:
    table = _get_table()
    table.put_item(Item={
        "PK": f"RMEM#{room_id}",
        "SK": f"{kb}#{slug}",
        "room_id": room_id,
        "slug": slug,
        "kb": kb,
        "relevance": _to_decimal(relevance),
    })


async def get_rooms(kb: str) -> list[dict]:
    table = _get_table()
    resp = table.query(
        IndexName="GSI1",
        KeyConditionExpression=Key("GSI1PK").eq(f"ROOMS#{kb}"),
    )
    items = [_strip_dynamo_keys(_clean_item(i)) for i in resp.get("Items", [])]
    items.sort(key=lambda x: x.get("article_count", 0), reverse=True)
    return items


async def get_room_members(room_id: int) -> list[dict]:
    table = _get_table()
    resp = table.query(KeyConditionExpression=Key("PK").eq(f"RMEM#{room_id}"))
    items = [_strip_dynamo_keys(_clean_item(i)) for i in resp.get("Items", [])]
    items.sort(key=lambda x: x.get("relevance", 0), reverse=True)
    return items


async def get_article_rooms(slug: str, kb: str) -> list[dict]:
    # Scan room members for this article (not ideal at scale, but fine for <10k rooms)
    table = _get_table()
    resp = table.scan(
        FilterExpression=Attr("PK").begins_with("RMEM#") & Attr("slug").eq(slug) & Attr("kb").eq(kb),
    )
    results = []
    for item in resp.get("Items", []):
        item = _clean_item(item)
        room_id = item.get("room_id")
        room = await _get_room_by_id(room_id)
        if room:
            results.append({
                "name": room["name"],
                "kb": room.get("kb", kb),
                "relevance": item.get("relevance", 1.0),
            })
    results.sort(key=lambda x: x.get("relevance", 0), reverse=True)
    return results


async def _get_room_by_id(room_id: int) -> dict | None:
    table = _get_table()
    resp = table.get_item(Key={"PK": f"ROOM#{room_id}", "SK": "META"})
    item = resp.get("Item")
    return _strip_dynamo_keys(_clean_item(item)) if item else None


async def clear_rooms(kb: str) -> None:
    table = _get_table()
    # Get all rooms for this KB
    rooms = await get_rooms(kb)
    with table.batch_writer() as batch:
        for room in rooms:
            room_id = room["id"]
            # Delete members
            members = table.query(KeyConditionExpression=Key("PK").eq(f"RMEM#{room_id}"))
            for m in members.get("Items", []):
                batch.delete_item(Key={"PK": m["PK"], "SK": m["SK"]})
            # Delete room
            batch.delete_item(Key={"PK": f"ROOM#{room_id}", "SK": "META"})


# ---------------------------------------------------------------------------
# Chat sessions
# ---------------------------------------------------------------------------

async def create_chat_session(session_id: str, title: str = "New Chat") -> dict:
    now = _now_iso()
    table = _get_table()
    table.put_item(Item={
        "PK": f"CHAT#{session_id}", "SK": "META",
        "id": session_id, "title": title,
        "created_at": now, "updated_at": now,
        "GSI1PK": "CHATS", "GSI1SK": f"{now}#{session_id}",
    })
    return {"id": session_id, "title": title, "created_at": now, "updated_at": now}


async def get_chat_sessions(limit: int = 30) -> list[dict]:
    table = _get_table()
    resp = table.query(
        IndexName="GSI1",
        KeyConditionExpression=Key("GSI1PK").eq("CHATS"),
        ScanIndexForward=False, Limit=limit,
    )
    return [_strip_dynamo_keys(_clean_item(i)) for i in resp.get("Items", [])]


async def get_chat_messages(session_id: str) -> list[dict]:
    table = _get_table()
    resp = table.query(
        KeyConditionExpression=Key("PK").eq(f"CHAT#{session_id}") & Key("SK").begins_with("MSG#"),
    )
    return [_strip_dynamo_keys(_clean_item(i)) for i in resp.get("Items", [])]


async def add_chat_message(session_id: str, role: str, content: str) -> dict:
    now = _now_iso()
    table = _get_table()
    msg_id = await _next_id("chatmsg")
    # Ensure session exists
    existing = table.get_item(Key={"PK": f"CHAT#{session_id}", "SK": "META"}).get("Item")
    if not existing:
        title = content[:60] if role == "user" else "New Chat"
        table.put_item(Item={
            "PK": f"CHAT#{session_id}", "SK": "META",
            "id": session_id, "title": title,
            "created_at": now, "updated_at": now,
            "GSI1PK": "CHATS", "GSI1SK": f"{now}#{session_id}",
        })
    else:
        table.update_item(
            Key={"PK": f"CHAT#{session_id}", "SK": "META"},
            UpdateExpression="SET updated_at = :u, GSI1SK = :gs",
            ExpressionAttributeValues={":u": now, ":gs": f"{now}#{session_id}"},
        )
        if role == "user" and existing.get("title") == "New Chat":
            table.update_item(
                Key={"PK": f"CHAT#{session_id}", "SK": "META"},
                UpdateExpression="SET title = :t",
                ExpressionAttributeValues={":t": content[:60]},
            )
    table.put_item(Item={
        "PK": f"CHAT#{session_id}", "SK": f"MSG#{now}#{msg_id}",
        "id": msg_id, "session_id": session_id,
        "role": role, "content": content, "created_at": now,
    })
    return {"id": msg_id, "session_id": session_id, "role": role, "content": content, "created_at": now}


async def delete_chat_session(session_id: str) -> None:
    table = _get_table()
    resp = table.query(KeyConditionExpression=Key("PK").eq(f"CHAT#{session_id}"))
    with table.batch_writer() as batch:
        for item in resp.get("Items", []):
            batch.delete_item(Key={"PK": item["PK"], "SK": item["SK"]})


# ---------------------------------------------------------------------------
# Chat events (observability)
# ---------------------------------------------------------------------------

async def log_chat_event(
    event: str, session_id: str | None = None, user_input: str | None = None,
    command: str | None = None, action: str | None = None,
    result: str | None = None, error: str | None = None,
) -> None:
    table = _get_table()
    evt_id = await _next_id("chatevt")
    now = _now_iso()
    item = {
        "PK": f"CHATEVT#{now[:10]}", "SK": f"{now}#{evt_id}",
        "id": evt_id, "event": event, "created_at": now,
        "GSI1PK": "CHATEVTS", "GSI1SK": f"{now}#{evt_id}",
    }
    if session_id: item["session_id"] = session_id
    if user_input: item["user_input"] = user_input[:200]
    if command: item["command"] = command
    if action: item["action"] = action
    if result: item["result"] = result
    if error: item["error"] = error
    table.put_item(Item=item)


async def get_chat_events(limit: int = 100, event_type: str | None = None) -> list[dict]:
    table = _get_table()
    resp = table.query(
        IndexName="GSI1",
        KeyConditionExpression=Key("GSI1PK").eq("CHATEVTS"),
        ScanIndexForward=False, Limit=limit,
    )
    items = [_strip_dynamo_keys(_clean_item(i)) for i in resp.get("Items", [])]
    if event_type:
        items = [i for i in items if i.get("event") == event_type]
    return items


async def get_chat_analytics() -> dict:
    events = await get_chat_events(limit=1000)
    event_counts = {}
    for e in events:
        evt = e.get("event", "unknown")
        event_counts[evt] = event_counts.get(evt, 0) + 1
    unmatched = {}
    for e in events:
        if e.get("event") == "command_unmatched" and e.get("user_input"):
            inp = e["user_input"]
            unmatched[inp] = unmatched.get(inp, 0) + 1
    sessions = await get_chat_sessions(limit=10000)
    return {
        "total_sessions": len(sessions),
        "total_messages": 0,
        "event_counts": event_counts,
        "unmatched_commands": [{"input": k, "count": v} for k, v in sorted(unmatched.items(), key=lambda x: -x[1])[:20]],
        "error_patterns": [],
    }


# ---------------------------------------------------------------------------
# Auto-discovery: Serper usage
# ---------------------------------------------------------------------------

def _today_utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


async def log_serper_call(
    query: str,
    num_results: int,
    kb: str | None,
    job_id: int | None,
) -> None:
    table = _get_table()
    serp_id = await _next_id("serper")
    day = _today_utc()
    now = _now_iso()
    item = {
        "PK": f"SERP#{day}",
        "SK": f"S#{now}#{serp_id:012d}",
        "id": serp_id,
        "day": day,
        "kb": kb or "",
        "job_id": job_id,
        "query": query,
        "num_results": int(num_results or 0),
        "created_at": now,
    }
    if kb:
        item["GSI1PK"] = f"SERPKB#{day}#{kb}"
        item["GSI1SK"] = f"{serp_id:012d}"
    table.put_item(Item=item)


async def serper_calls_today(kb: str) -> int:
    table = _get_table()
    day = _today_utc()
    resp = table.query(
        IndexName="GSI1",
        KeyConditionExpression=Key("GSI1PK").eq(f"SERPKB#{day}#{kb}"),
        Select="COUNT",
    )
    return int(resp.get("Count", 0))


# ---------------------------------------------------------------------------
# Auto-discovery: topic candidates
# ---------------------------------------------------------------------------


def _topic_key(topic: str) -> str:
    """Stable key for the (kb, topic) UNIQUE constraint."""
    return topic.strip().lower()


async def insert_topic_candidate(
    kb: str,
    topic: str,
    source: str,
    source_ref: str | None,
    score: float = 0.0,
) -> bool:
    table = _get_table()
    cand_id = await _next_id("candidate")
    now = _now_iso()
    sk = f"TC#{_topic_key(topic)}"
    # Inverse-score key so DESC sort works as default ASC on GSI1
    inv = f"{1_000_000 - int(float(score) * 1000):010d}"
    try:
        table.put_item(
            Item={
                "PK": f"TC#{kb}",
                "SK": sk,
                "id": cand_id,
                "kb": kb,
                "topic": topic,
                "source": source,
                "source_ref": source_ref,
                "score": _to_decimal(float(score)),
                "status": "pending",
                "reason": None,
                "created_at": now,
                "enqueued_at": None,
                "job_id": None,
                "GSI1PK": f"TCSTATUS#{kb}#pending",
                "GSI1SK": f"{inv}#{now}#{cand_id:012d}",
            },
            ConditionExpression="attribute_not_exists(PK) AND attribute_not_exists(SK)",
        )
        return True
    except table.meta.client.exceptions.ConditionalCheckFailedException:
        return False


async def get_pending_candidates(kb: str, limit: int) -> list[dict]:
    if limit <= 0:
        return []
    table = _get_table()
    resp = table.query(
        IndexName="GSI1",
        KeyConditionExpression=Key("GSI1PK").eq(f"TCSTATUS#{kb}#pending"),
        Limit=limit,
    )
    return [_strip_dynamo_keys(_clean_item(i)) for i in resp.get("Items", [])]


async def _find_candidate_keys(candidate_id: int) -> tuple[str, str] | None:
    """Locate a topic_candidate row by id via the GSI1 (per-status) projections."""
    table = _get_table()
    for status in ("pending", "enqueued", "skipped"):
        # Heuristic: scan within the small per-status partitions of every KB
        # via a Scan filter. Counts are tiny (low thousands across all KBs).
        resp = table.scan(
            FilterExpression=Attr("PK").begins_with("TC#") & Attr("id").eq(int(candidate_id)) & Attr("status").eq(status),
        )
        items = resp.get("Items", [])
        if items:
            return items[0]["PK"], items[0]["SK"]
    return None


async def mark_candidate_enqueued(candidate_id: int, job_id: int) -> None:
    table = _get_table()
    keys = await _find_candidate_keys(candidate_id)
    if not keys:
        return
    pk, sk = keys
    now = _now_iso()
    table.update_item(
        Key={"PK": pk, "SK": sk},
        UpdateExpression=(
            "SET #s = :s, enqueued_at = :ea, job_id = :jid, "
            "GSI1PK = :gpk REMOVE GSI1SK"
        ),
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={
            ":s": "enqueued",
            ":ea": now,
            ":jid": int(job_id),
            ":gpk": "TCSTATUS#enqueued",
        },
    )


async def mark_candidate_skipped(candidate_id: int, reason: str) -> None:
    table = _get_table()
    keys = await _find_candidate_keys(candidate_id)
    if not keys:
        return
    pk, sk = keys
    table.update_item(
        Key={"PK": pk, "SK": sk},
        UpdateExpression=(
            "SET #s = :s, reason = :r, GSI1PK = :gpk REMOVE GSI1SK"
        ),
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={
            ":s": "skipped",
            ":r": reason,
            ":gpk": "TCSTATUS#skipped",
        },
    )


async def count_pending_candidates(kb: str) -> int:
    table = _get_table()
    resp = table.query(
        IndexName="GSI1",
        KeyConditionExpression=Key("GSI1PK").eq(f"TCSTATUS#{kb}#pending"),
        Select="COUNT",
    )
    return int(resp.get("Count", 0))


# ---------------------------------------------------------------------------
# Auto-discovery: per-KB config
# ---------------------------------------------------------------------------


_AUTO_DISCOVERY_DEFAULTS = {
    "enabled": 0,
    "daily_budget": 500,
    "max_per_hour": 3,
    "strategy": "hybrid",
    "seed_topics": None,
    "llm_sample": 5,
}


async def get_auto_discovery_config(kb: str) -> dict | None:
    table = _get_table()
    resp = table.get_item(Key={"PK": "ADCFG", "SK": f"CFG#{kb}"})
    item = resp.get("Item")
    if not item:
        return None
    return _strip_dynamo_keys(_clean_item(item))


async def list_enabled_auto_discovery_configs() -> list[dict]:
    table = _get_table()
    resp = table.query(
        IndexName="GSI1",
        KeyConditionExpression=Key("GSI1PK").eq("ADCFGENABLED"),
    )
    return [_strip_dynamo_keys(_clean_item(i)) for i in resp.get("Items", [])]


async def upsert_auto_discovery_config(kb: str, **fields) -> dict:
    import json as _json
    allowed = set(_AUTO_DISCOVERY_DEFAULTS.keys())
    clean = {k: v for k, v in fields.items() if k in allowed}
    if "enabled" in clean:
        clean["enabled"] = 1 if clean["enabled"] else 0
    if "seed_topics" in clean and clean["seed_topics"] is not None and not isinstance(clean["seed_topics"], str):
        clean["seed_topics"] = _json.dumps(clean["seed_topics"])

    table = _get_table()
    existing = (await get_auto_discovery_config(kb)) or {}
    merged = {**_AUTO_DISCOVERY_DEFAULTS, **existing, **clean, "kb": kb, "updated_at": _now_iso()}
    item = {
        "PK": "ADCFG",
        "SK": f"CFG#{kb}",
        **merged,
    }
    if int(merged.get("enabled", 0)) == 1:
        item["GSI1PK"] = "ADCFGENABLED"
        item["GSI1SK"] = kb
    # Cast numeric fields
    for k in ("daily_budget", "max_per_hour", "llm_sample", "enabled"):
        if k in item and item[k] is not None:
            item[k] = int(item[k])
    table.put_item(Item=item)
    return _strip_dynamo_keys(_clean_item(item))


# ---------------------------------------------------------------------------
# Article version snapshots
# ---------------------------------------------------------------------------

import hashlib as _hashlib


async def save_article_version(
    kb: str,
    article_slug: str,
    full_content: str,
    *,
    job_id: int | None = None,
    change_type: str = "updated",
    note: str | None = None,
) -> int:
    table = _get_table()
    version_id = await _next_id("version")
    now = _now_iso()
    content_hash = _hashlib.sha256(full_content.encode("utf-8", errors="replace")).hexdigest()
    table.put_item(Item={
        "PK": f"AV#{kb}#{article_slug}",
        "SK": f"V#{now}#{version_id:012d}",
        "id": version_id,
        "kb": kb,
        "article_slug": article_slug,
        "full_content": full_content,
        "content_hash": content_hash,
        "change_type": change_type,
        "job_id": job_id,
        "note": note,
        "created_at": now,
        # GSI2 lets us look up by version_id alone (for the diff endpoint)
        "GSI2PK": "AVID",
        "GSI2SK": f"{version_id:012d}",
    })
    return version_id


async def get_article_versions(kb: str, article_slug: str, limit: int = 50) -> list[dict]:
    table = _get_table()
    resp = table.query(
        KeyConditionExpression=Key("PK").eq(f"AV#{kb}#{article_slug}") & Key("SK").begins_with("V#"),
        ScanIndexForward=False,
        Limit=limit,
    )
    return [_strip_dynamo_keys(_clean_item(i)) for i in resp.get("Items", [])]


async def get_article_at_timestamp(
    kb: str, article_slug: str, before_iso: str,
) -> dict | None:
    table = _get_table()
    resp = table.query(
        KeyConditionExpression=Key("PK").eq(f"AV#{kb}#{article_slug}") & Key("SK").lte(f"V#{before_iso}~"),
        ScanIndexForward=False,
        Limit=1,
    )
    items = resp.get("Items", [])
    if not items:
        return None
    return _strip_dynamo_keys(_clean_item(items[0]))


async def get_article_version_by_id(version_id: int) -> dict | None:
    table = _get_table()
    resp = table.query(
        IndexName="GSI2",
        KeyConditionExpression=Key("GSI2PK").eq("AVID") & Key("GSI2SK").eq(f"{int(version_id):012d}"),
        Limit=1,
    )
    items = resp.get("Items", [])
    if not items:
        return None
    return _strip_dynamo_keys(_clean_item(items[0]))


# ---------------------------------------------------------------------------
# Article claims (structured fact storage)
# ---------------------------------------------------------------------------


def _claim_hash(claim_text: str) -> str:
    return _hashlib.sha256(claim_text.encode("utf-8", errors="replace")).hexdigest()[:32]


async def save_claim(
    kb: str,
    article_slug: str,
    claim_text: str,
    *,
    claim_type: str = "general",
    sources_json: str | None = None,
    confidence: float = 0.5,
    status: str = "unverified",
) -> int:
    table = _get_table()
    pk = f"CLAIM#{kb}#{article_slug}"
    sk = f"CLAIM#{_claim_hash(claim_text)}"
    now = _now_iso()
    existing = table.get_item(Key={"PK": pk, "SK": sk}).get("Item")
    if existing:
        table.update_item(
            Key={"PK": pk, "SK": sk},
            UpdateExpression=(
                "SET claim_type = :ct, sources_json = :sj, confidence = :c, "
                "#s = :st, updated_at = :u"
            ),
            ExpressionAttributeNames={"#s": "status"},
            ExpressionAttributeValues={
                ":ct": claim_type,
                ":sj": sources_json,
                ":c": _to_decimal(float(confidence)),
                ":st": status,
                ":u": now,
            },
        )
        return int(existing.get("id", 0))

    claim_id = await _next_id("claim")
    table.put_item(Item={
        "PK": pk,
        "SK": sk,
        "id": claim_id,
        "kb": kb,
        "article_slug": article_slug,
        "claim_text": claim_text,
        "claim_type": claim_type,
        "sources_json": sources_json,
        "confidence": _to_decimal(float(confidence)),
        "status": status,
        "last_checked_at": None,
        "created_at": now,
        "updated_at": now,
        # GSI1 supports the periodic stale-claims audit
        "GSI1PK": "CLAIMAUDIT",
        "GSI1SK": f"NEVER#{claim_id:012d}",
    })
    return claim_id


async def get_claims_for_article(kb: str, article_slug: str) -> list[dict]:
    table = _get_table()
    resp = table.query(
        KeyConditionExpression=Key("PK").eq(f"CLAIM#{kb}#{article_slug}"),
    )
    items = [_strip_dynamo_keys(_clean_item(i)) for i in resp.get("Items", [])]
    items.sort(key=lambda r: int(r.get("id", 0)))
    return items


async def get_stale_claims(days: int = 90) -> list[dict]:
    cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
    table = _get_table()
    resp = table.query(
        IndexName="GSI1",
        KeyConditionExpression=Key("GSI1PK").eq("CLAIMAUDIT"),
        Limit=500,
    )
    out: list[dict] = []
    for raw in resp.get("Items", []):
        item = _strip_dynamo_keys(_clean_item(raw))
        last = item.get("last_checked_at")
        if last is None or (isinstance(last, str) and last < cutoff):
            out.append(item)
    return out[:500]


async def update_claim_status(claim_id: int, status: str, confidence: float) -> None:
    table = _get_table()
    # Locate the claim row by id (GSI1 audit projection contains all claims).
    resp = table.query(
        IndexName="GSI1",
        KeyConditionExpression=Key("GSI1PK").eq("CLAIMAUDIT"),
        FilterExpression=Attr("id").eq(int(claim_id)),
        Limit=10,
    )
    items = resp.get("Items", [])
    if not items:
        return
    target = items[0]
    now = _now_iso()
    table.update_item(
        Key={"PK": target["PK"], "SK": target["SK"]},
        UpdateExpression=(
            "SET #s = :s, confidence = :c, last_checked_at = :lc, updated_at = :u, "
            "GSI1SK = :gs"
        ),
        ExpressionAttributeNames={"#s": "status"},
        ExpressionAttributeValues={
            ":s": status,
            ":c": _to_decimal(float(confidence)),
            ":lc": now,
            ":u": now,
            ":gs": f"{now}#{int(claim_id):012d}",
        },
    )


# ---------------------------------------------------------------------------
# Per-KB settings
# ---------------------------------------------------------------------------


_KB_SETTINGS_FIELDS = (
    "synthesis_provider", "synthesis_model",
    "query_provider", "query_model", "persona",
    # Agent memory — episodic log + per-domain source reliability.
    # Stored as JSON text so we don't fracture the schema further.
    "agent_episodes", "source_reliability",
)


async def get_kb_settings(kb: str) -> dict | None:
    table = _get_table()
    resp = table.get_item(Key={"PK": "KBSET", "SK": f"KB#{kb}"})
    item = resp.get("Item")
    if not item:
        return None
    return _strip_dynamo_keys(_clean_item(item))


async def upsert_kb_settings(kb: str, **fields) -> dict:
    clean = {k: v for k, v in fields.items() if k in _KB_SETTINGS_FIELDS}
    for k in list(clean.keys()):
        if isinstance(clean[k], str) and clean[k].strip() == "":
            clean[k] = None

    existing = await get_kb_settings(kb) or {}
    merged: dict = {f: existing.get(f) for f in _KB_SETTINGS_FIELDS}
    merged.update(clean)
    merged["kb"] = kb
    merged["updated_at"] = _now_iso()

    table = _get_table()
    table.put_item(Item={"PK": "KBSET", "SK": f"KB#{kb}", **merged})
    return merged


# ---------------------------------------------------------------------------
# Article metadata index
#
# Single-table design: one item per article, keyed on
#   PK = ARTMETA#<kb>
#   SK = <slug>
#
# Every ``wiki.create_article`` / ``update_article`` / ``delete_article``
# writes through here so ``list_article_metas(kb)`` can answer the
# "give me every article in this KB" query in a single DynamoDB query()
# instead of iterating the whole S3 prefix. Body text still lives in S3
# — only the fields the list views care about land in DynamoDB.
# ---------------------------------------------------------------------------


async def upsert_article_meta(kb: str, slug: str, meta: dict) -> None:
    """Write-through an article's metadata into DynamoDB.

    Stores everything from ``_normalize_article_meta`` plus the
    derived quality signals (h2_count, link_count, code_count,
    wikilink_count, has_tech_tag) so shallow-article and quality
    scans can read them from the index without re-fetching bodies.
    """
    if not kb or not slug:
        return
    table = _get_table()
    item: dict = {
        "PK": f"ARTMETA#{kb}",
        "SK": slug,
        "kb": kb,
        "slug": slug,
        "title": str(meta.get("title") or slug),
        "summary": str(meta.get("summary") or ""),
        "tags": list(meta.get("tags") or []) or None,
        "status": str(meta.get("status") or "draft"),
        "confidence": str(meta.get("confidence") or "medium"),
        "updated": str(meta.get("updated") or ""),
        "source_type": str(meta.get("source_type") or "manual"),
        "source_files": list(meta.get("source_files") or []) or None,
        "word_count": int(meta.get("word_count") or 0),
        "h2_count": int(meta.get("h2_count") or 0),
        "link_count": int(meta.get("link_count") or 0),
        "code_count": int(meta.get("code_count") or 0),
        "wikilink_count": int(meta.get("wikilink_count") or 0),
        "has_tech_tag": bool(meta.get("has_tech_tag") or False),
        "indexed_at": _now_iso(),
    }
    table.put_item(Item={k: v for k, v in item.items() if v is not None})


async def get_article_meta(kb: str, slug: str) -> dict | None:
    table = _get_table()
    resp = table.get_item(Key={"PK": f"ARTMETA#{kb}", "SK": slug})
    item = resp.get("Item")
    if not item:
        return None
    return _strip_dynamo_keys(_clean_item(item))


async def list_article_metas(kb: str) -> list[dict]:
    """Return every article meta row for a KB, newest-first by ``updated``.

    Auto-paginates via ExclusiveStartKey so large KBs still return the
    full set in one call.
    """
    table = _get_table()
    items: list[dict] = []
    last_key = None
    while True:
        query_kwargs = {
            "KeyConditionExpression": Key("PK").eq(f"ARTMETA#{kb}"),
        }
        if last_key is not None:
            query_kwargs["ExclusiveStartKey"] = last_key
        resp = table.query(**query_kwargs)
        items.extend(resp.get("Items", []))
        last_key = resp.get("LastEvaluatedKey")
        if not last_key:
            break
    rows = [_strip_dynamo_keys(_clean_item(i)) for i in items]
    # Newest-first, falling back to slug order for rows without ``updated``.
    rows.sort(key=lambda r: (r.get("updated") or "", r.get("slug") or ""), reverse=True)
    return rows


async def delete_article_meta(kb: str, slug: str) -> None:
    if not kb or not slug:
        return
    table = _get_table()
    table.delete_item(Key={"PK": f"ARTMETA#{kb}", "SK": slug})


async def article_meta_count(kb: str) -> int:
    """Return how many metadata rows exist for a KB (cheap — no item reads)."""
    table = _get_table()
    total = 0
    last_key = None
    while True:
        query_kwargs = {
            "KeyConditionExpression": Key("PK").eq(f"ARTMETA#{kb}"),
            "Select": "COUNT",
        }
        if last_key is not None:
            query_kwargs["ExclusiveStartKey"] = last_key
        resp = table.query(**query_kwargs)
        total += int(resp.get("Count", 0))
        last_key = resp.get("LastEvaluatedKey")
        if not last_key:
            break
    return total


# ---------------------------------------------------------------------------
# LLM usage totals (all-time)
# ---------------------------------------------------------------------------


async def record_llm_usage_total(
    provider: str, model: str, kind: str,
    *, input_tokens: int, output_tokens: int,
) -> None:
    """Atomic counter update: PK=USAGE, SK=provider#model#kind."""
    table = _get_table()
    now = _now_iso()
    key = {"PK": "USAGE", "SK": f"{provider}#{model}#{kind}"}
    try:
        table.update_item(
            Key=key,
            UpdateExpression=(
                "ADD calls :one, input_tokens :in_tok, output_tokens :out_tok "
                "SET last_seen = :now, "
                "provider = if_not_exists(provider, :p), "
                "model = if_not_exists(model, :m), "
                "kind = if_not_exists(kind, :k), "
                "first_seen = if_not_exists(first_seen, :now)"
            ),
            ExpressionAttributeValues={
                ":one": 1,
                ":in_tok": int(input_tokens or 0),
                ":out_tok": int(output_tokens or 0),
                ":now": now,
                ":p": provider,
                ":m": model,
                ":k": kind,
            },
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("record_llm_usage_total failed: %s", exc)


async def get_llm_usage_totals() -> list[dict]:
    """Scan every lifetime usage row under PK=USAGE."""
    table = _get_table()
    rows: list[dict] = []
    last_key = None
    while True:
        query_kwargs = {"KeyConditionExpression": Key("PK").eq("USAGE")}
        if last_key is not None:
            query_kwargs["ExclusiveStartKey"] = last_key
        resp = table.query(**query_kwargs)
        for item in resp.get("Items", []):
            rows.append({
                "provider": item.get("provider", ""),
                "model": item.get("model", ""),
                "kind": item.get("kind", ""),
                "calls": int(item.get("calls", 0)),
                "input_tokens": int(item.get("input_tokens", 0)),
                "output_tokens": int(item.get("output_tokens", 0)),
                "first_seen": item.get("first_seen", ""),
                "last_seen": item.get("last_seen", ""),
            })
        last_key = resp.get("LastEvaluatedKey")
        if not last_key:
            break
    rows.sort(key=lambda r: r["calls"], reverse=True)
    return rows


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _strip_dynamo_keys(item: dict) -> dict:
    """Remove DynamoDB-specific keys from an item for API compatibility."""
    for key in ("PK", "SK", "GSI1PK", "GSI1SK", "GSI2PK", "GSI2SK"):
        item.pop(key, None)
    return item

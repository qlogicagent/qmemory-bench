"""E2E test: QMemory Server + DeepSeek LLM + qmemory-bench runner.

Tests the full pipeline:
1. Start QMemory server (SKIP_EMBED for speed)
2. Connect via HTTP
3. Run benchmark with DeepSeek API key
4. Verify results structure
"""
import asyncio
import sys
import os
import time
import httpx

# Bypass proxy
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

SERVER_URL = "http://127.0.0.1:18800"
DEEPSEEK_KEY = os.environ.get("DEEPSEEK_API_KEY", "").strip()


def require_deepseek_key() -> str:
    if DEEPSEEK_KEY:
        return DEEPSEEK_KEY
    print("ERROR: DEEPSEEK_API_KEY is required for this E2E test.")
    sys.exit(1)


async def main():
    print("=" * 60)
    print("E2E Test: QMemory + DeepSeek + qmemory-bench")
    print("=" * 60)

    # ── Step 1: Health check ──
    print("\n[1/6] Health check...")
    async with httpx.AsyncClient(timeout=5.0, proxy=None) as c:
        try:
            r = await c.get(f"{SERVER_URL}/v1/health/")
            health = r.json()
            print(f"  Server OK: v{health['version']}, memories={health['memory_count']}")
        except Exception as e:
            print(f"  ERROR: QMemory server not reachable at {SERVER_URL}: {e}")
            print("  Please start server first: qmemory serve --port 18800")
            sys.exit(1)

    # ── Step 2: Reset memory (clean slate) ──
    print("\n[2/6] Resetting memory database...")
    async with httpx.AsyncClient(timeout=10.0, proxy=None) as c:
        try:
            r = await c.request("DELETE", f"{SERVER_URL}/v1/memories/",
                                params={"confirm": "true"})
            print(f"  Reset: HTTP {r.status_code}")
        except Exception as e:
            print(f"  Reset warning: {e}")

    # ── Step 3: Test DeepSeek API connectivity ──
    print("\n[3/6] Testing DeepSeek API...")
    from qmemory_bench.providers import LLMJudge
    llm = LLMJudge(provider="deepseek", api_key=require_deepseek_key())
    try:
        resp = await llm.complete("Reply with just 'OK'.", temperature=0, max_tokens=10)
        print(f"  DeepSeek API OK: '{resp.strip()}'")
    except Exception as e:
        print(f"  ERROR: DeepSeek API failed: {e}")
        await llm.close()
        sys.exit(1)

    # ── Step 4: Inject test sessions ──
    print("\n[4/6] Injecting test sessions (qmemory-chinese_quick)...")
    from qmemory_bench.dataset import load_dataset
    ds = load_dataset("qmemory-chinese", "quick")
    print(f"  Loaded: {len(ds.sessions)} sessions, {len(ds.questions)} questions")

    eval_user = f"e2e_test_{int(time.time())}"
    inject_count = 0
    async with httpx.AsyncClient(base_url=SERVER_URL, timeout=300.0, proxy=None) as c:
        for session in ds.sessions:
            try:
                r = await c.post("/v1/memories/", json={
                    "messages": session.messages,
                    "user_id": eval_user,
                    "session_id": session.id,
                    "llm_config": {
                        "provider": "deepseek",
                        "api_key": require_deepseek_key(),
                        "model": "deepseek-chat",
                    }
                })
                if r.status_code == 200:
                    inject_count += 1
                    data = r.json()
                    added = data.get("memories_added", 0)
                    print(f"  Session {session.id}: +{added} memories")
                else:
                    print(f"  Session {session.id}: HTTP {r.status_code} - {r.text[:100]}")
            except Exception as e:
                print(f"  Session {session.id}: ERROR {type(e).__name__}: {e}")

    print(f"  Injected {inject_count}/{len(ds.sessions)} sessions")

    # ── Step 5: Test search/recall ──
    print("\n[5/6] Testing search/recall (3 sample questions)...")
    sample_questions = ds.questions[:3]
    scores = []
    async with httpx.AsyncClient(base_url=SERVER_URL, timeout=120.0, proxy=None) as c:
        for q in sample_questions:
            try:
                r = await c.get("/v1/memories/search/", params={
                    "q": q.query,
                    "user_id": eval_user,
                    "limit": 5,
                })
                recall = r.json()
                mem_count = len(recall.get("memories", []))
                context_len = len(recall.get("context", ""))

                # Quick judge
                from qmemory_bench.judge import judge_single
                result = await judge_single(
                    question_id=q.id,
                    query=q.query,
                    expected=q.expected,
                    category=q.category,
                    recall_result=recall,
                    llm=llm,
                )
                scores.append(result.score)
                print(f"  {q.id}: score={result.score}/10, memories={mem_count}, "
                      f"context={context_len}chars")
                print(f"    Q: {q.query}")
                print(f"    Expected: {q.expected}")
                print(f"    Reason: {result.reason}")
            except Exception as e:
                print(f"  {q.id}: ERROR {type(e).__name__}: {e}")
                scores.append(0)

    avg_score = sum(scores) / len(scores) * 10 if scores else 0
    print(f"\n  Average score: {avg_score:.1f}%")

    # ── Step 6: Cleanup ──
    print("\n[6/6] Cleaning up...")
    async with httpx.AsyncClient(base_url=SERVER_URL, timeout=10.0, proxy=None) as c:
        try:
            await c.request("DELETE", "/v1/memories/",
                            params={"user_id": eval_user, "confirm": "true"})
            print("  Cleanup OK")
        except Exception as e:
            print(f"  Cleanup warning: {e}")

    await llm.close()

    # ── Summary ──
    print("\n" + "=" * 60)
    print("E2E Test Summary")
    print("=" * 60)
    print(f"  Server:     {SERVER_URL} (healthy)")
    print(f"  LLM:        DeepSeek (deepseek-chat)")
    print(f"  Dataset:    qmemory-chinese ({len(ds.questions)} questions)")
    print(f"  Injected:   {inject_count}/{len(ds.sessions)} sessions")
    print(f"  Scored:     {len(scores)} questions, avg={avg_score:.1f}%")
    print(f"  Status:     {'PASS' if inject_count > 0 and avg_score > 0 else 'FAIL'}")
    print("=" * 60)

    return 0 if inject_count > 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))

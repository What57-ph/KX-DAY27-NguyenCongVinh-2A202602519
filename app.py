"""Exercise 5 — Streamlit approval UI for the HITL PR review agent.

Run with:
    uv run streamlit run app.py

Goal: wrap the LangGraph built in exercises 1–4 in a web UI that adapts to
the confidence bucket of each PR.

Routing thresholds (common/schemas.py):
    > 72%        auto_approve     UI shows a success card; reviewer does nothing
    58 – 72%     human_approval   UI shows Approve / Reject / Edit buttons
    <  58%       escalate         UI shows a question form for the reviewer
"""

from __future__ import annotations

import asyncio
import uuid

import streamlit as st
from dotenv import load_dotenv
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command

from common.db import db_conn, db_path
from exercises.exercise_4_audit import build_graph


load_dotenv()


# ─── Session state ─────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = None
if "pr_url" not in st.session_state:
    st.session_state.pr_url = ""
if "interrupts" not in st.session_state:
    st.session_state.interrupts = []
if "final" not in st.session_state:
    st.session_state.final = None
if "alternate_final" not in st.session_state:
    st.session_state.alternate_final = None


# ─── Page setup ────────────────────────────────────────────────────────────
st.set_page_config(page_title="HITL PR Review", layout="wide")
st.title("HITL PR Review Agent")


# ─── Sidebar — recent sessions ─────────────────────────────────────────────
with st.sidebar:
    st.header("Recent sessions")

    async def dashboard_data() -> tuple[list[dict], dict]:
        async with db_conn() as conn:
            async with conn.execute(
                """
                SELECT thread_id,
                       pr_url,
                       MAX(risk_level) AS worst_risk,
                       MAX(timestamp)  AS last_event
                  FROM audit_events
                 GROUP BY thread_id, pr_url
                 ORDER BY MAX(timestamp) DESC
                 LIMIT 25
                """
            ) as cur:
                sessions = [dict(row) for row in await cur.fetchall()]
            async with conn.execute(
                """
                WITH analyses AS (
                    SELECT thread_id, AVG(confidence) AS confidence
                      FROM audit_events
                     WHERE action = 'analyze'
                     GROUP BY thread_id
                ), choices AS (
                    SELECT thread_id, decision = 'approve' AS approved
                      FROM audit_events
                     WHERE action = 'human_approval'
                       AND decision IN ('approve', 'reject', 'edit')
                )
                SELECT COUNT(*) AS sessions,
                       AVG(analyses.confidence) AS confidence,
                       AVG(choices.approved) AS approval_rate
                  FROM choices JOIN analyses USING (thread_id)
                """
            ) as cur:
                calibration = dict(await cur.fetchone())
        return sessions, calibration

    sessions, calibration = asyncio.run(dashboard_data())
    if sessions:
        selection = st.dataframe(
            sessions,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
        )
        if selection.selection.rows:
            selected = sessions[selection.selection.rows[0]]
            if st.session_state.thread_id != selected["thread_id"]:
                st.session_state.thread_id = selected["thread_id"]
                st.session_state.pr_url = selected["pr_url"]
                st.rerun()
    else:
        st.caption("No review sessions yet")

    st.subheader("Confidence calibration")
    if calibration["sessions"]:
        confidence = calibration["confidence"]
        approval_rate = calibration["approval_rate"]
        st.metric("Model confidence", f"{confidence:.0%}")
        st.metric("Human approval rate", f"{approval_rate:.0%}")
        if confidence > approval_rate:
            st.warning("Model is over-confident")
        elif confidence < approval_rate:
            st.info("Model is under-confident")
        else:
            st.success("Model is calibrated")
        st.caption(f"Based on {calibration['sessions']} human-reviewed session(s)")
    else:
        st.caption("Needs at least one human-reviewed session")


# ─── Top form — start a new review ─────────────────────────────────────────
with st.form("start"):
    pr_url = st.text_input(
        "PR URL", value=st.session_state.pr_url,
        placeholder="https://github.com/VinUni-AI20k/PR-Demo/pull/1",
    )
    submitted = st.form_submit_button("Run review")


# ─── Renderers per interrupt kind ──────────────────────────────────────────
def render_approval_card(payload: dict) -> dict | None:
    """58–72% bucket: show the LLM review + 3 buttons. Return resume dict or None."""
    conf = payload["confidence"]
    st.subheader(f"Approval requested — confidence {conf:.0%}")
    st.caption(payload["confidence_reasoning"])
    st.markdown(payload["summary"])

    for c in payload.get("comments", []):
        st.markdown(f"- **[{c['severity']}]** `{c['file']}:{c.get('line') or '?'}` — {c['body']}")

    with st.expander("Diff"):
        st.code(payload.get("diff_preview", ""), language="diff")

    feedback = st.text_input("Feedback (optional)", key="approval_feedback")
    col1, col2, col3 = st.columns(3)
    if col1.button("Approve", type="primary"):
        return {"choice": "approve", "feedback": feedback}
    if col2.button("Reject"):
        return {"choice": "reject", "feedback": feedback}
    if col3.button("Edit"):
        return {"choice": "edit", "feedback": feedback}
    return None


def render_escalation_cards(interrupts) -> dict | None:
    """Render every parallel reviewer request and resume all interrupt IDs together."""
    with st.form("escalations"):
        responses = {}
        for interrupt in interrupts:
            payload = interrupt.value
            reviewer = payload.get("reviewer", "Reviewer")
            st.subheader(f"{reviewer} · confidence {payload['confidence']:.0%}")
            st.caption(payload["confidence_reasoning"])
            if payload.get("risk_factors"):
                st.error("Risks: " + ", ".join(payload["risk_factors"]))
            st.markdown(payload["summary"])
            responses[interrupt.id] = {
                question: st.text_input(question, key=f"{interrupt.id}:{i}")
                for i, question in enumerate(payload["questions"])
            }
        if st.form_submit_button("Submit all answers"):
            return responses
    return None


# ─── Drive the graph ───────────────────────────────────────────────────────
async def run_graph(pr_url: str, thread_id: str, resume_value=None):
    """Invoke the graph once. Returns the final result or {'__interrupt__': ...}."""
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        await cp.setup()
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}
        if resume_value is None:
            result = await app.ainvoke({"pr_url": pr_url, "thread_id": thread_id}, cfg)
        else:
            result = await app.ainvoke(Command(resume=resume_value), cfg)
        return result


async def state_history(thread_id: str):
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        app = build_graph(cp)
        cfg = {"configurable": {"thread_id": thread_id}}
        return [snapshot async for snapshot in app.aget_state_history(cfg)]


async def resume_checkpoint(checkpoint_config: dict, resume_value):
    async with AsyncSqliteSaver.from_conn_string(db_path()) as cp:
        app = build_graph(cp)
        return await app.ainvoke(Command(resume=resume_value), checkpoint_config)


def save_result(result) -> None:
    if "__interrupt__" in result:
        st.session_state.interrupts = list(result["__interrupt__"])
    else:
        st.session_state.interrupts = []
        st.session_state.final = result


# ─── Main flow ─────────────────────────────────────────────────────────────
if submitted and pr_url:
    st.session_state.pr_url = pr_url
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.interrupts = []
    st.session_state.final = None
    st.session_state.alternate_final = None

    with st.spinner("Fetching PR + asking the LLM..."):
        result = asyncio.run(run_graph(pr_url, st.session_state.thread_id))

    save_result(result)

# Render the current interrupt card(s), if any
interrupts = st.session_state.interrupts
if interrupts:
    answer = (
        render_approval_card(interrupts[0].value)
        if len(interrupts) == 1 and interrupts[0].value["kind"] == "approval_request"
        else render_escalation_cards(interrupts)
    )
    if answer is not None:
        with st.spinner("Resuming..."):
            result = asyncio.run(run_graph(
                st.session_state.pr_url, st.session_state.thread_id, resume_value=answer,
            ))
        save_result(result)
        st.rerun()

# Render final state, if reached
if st.session_state.final is not None:
    final = st.session_state.final
    action = final.get("final_action", "?")
    if action.startswith("auto") or action.startswith("committed"):
        st.success(f"✓ {action} — comment posted to {st.session_state.pr_url}")
    elif action == "rejected":
        st.warning("Rejected — no comment posted")
    else:
        st.info(f"final_action = {action}")
    st.caption(f"thread_id = {st.session_state.thread_id}  ·  replay: "
               f"`uv run python -m audit.replay --thread {st.session_state.thread_id}`")

    with st.expander("Time-travel: retry an earlier human decision"):
        history = asyncio.run(state_history(st.session_state.thread_id))
        checkpoints = [
            (snapshot, [item for task in snapshot.tasks for item in task.interrupts])
            for snapshot in history
            if any(task.interrupts for task in snapshot.tasks)
        ]
        if not checkpoints:
            st.caption("This session has no human-interrupt checkpoint")
        else:
            selected_index = st.selectbox(
                "Checkpoint",
                range(len(checkpoints)),
                format_func=lambda i: (
                    f"{checkpoints[i][0].created_at} · "
                    f"{', '.join(checkpoints[i][0].next)}"
                ),
            )
            snapshot, old_interrupts = checkpoints[selected_index]
            with st.form("time_travel"):
                alternate_answers = {}
                for old_interrupt in old_interrupts:
                    old_payload = old_interrupt.value
                    if old_payload["kind"] == "approval_request":
                        choice = st.selectbox(
                            "Different decision", ("approve", "reject", "edit"),
                            key=f"travel-choice-{old_interrupt.id}",
                        )
                        feedback = st.text_input(
                            "Feedback", key=f"travel-feedback-{old_interrupt.id}",
                        )
                        alternate_answers[old_interrupt.id] = {
                            "choice": choice, "feedback": feedback,
                        }
                    else:
                        reviewer = old_payload.get("reviewer", "Reviewer")
                        st.markdown(f"**{reviewer}**")
                        alternate_answers[old_interrupt.id] = {
                            question: st.text_input(
                                question, key=f"travel-{old_interrupt.id}-{i}",
                            )
                            for i, question in enumerate(old_payload["questions"])
                        }
                travel = st.form_submit_button("Run alternate branch")

            if travel:
                with st.spinner("Running from the selected checkpoint..."):
                    alternate = asyncio.run(resume_checkpoint(snapshot.config, alternate_answers))
                st.session_state.alternate_final = alternate

            if st.session_state.alternate_final is not None:
                alternate = st.session_state.alternate_final
                col1, col2 = st.columns(2)
                col1.metric("Original outcome", final.get("final_action", "?"))
                col2.metric("Alternate outcome", alternate.get("final_action", "interrupted"))

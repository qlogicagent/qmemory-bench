"""NiceGUI-based Benchmark UI — full-featured evaluation desktop.

Features:
  - Connection status indicator (green/red dot, auto-poll)
  - Single-run & comparison modes (compare multiple LLMs side-by-side)
  - Real-time progress display (session injection, question evaluation)
  - Chinese / English i18n (default Chinese)
  - Export: download JSON report / copy text to clipboard
  - Dark / light theme toggle
  - Memory reset with confirmation dialog

Launch: qmemory-bench ui [--port 8090]
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def launch_ui(port: int = 8090, default_target: str = "http://localhost:18800"):
    """Launch the NiceGUI benchmark UI."""
    # Disable system proxy for local connections (httpx proxy=None still uses env vars)
    for k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        os.environ.pop(k, None)
    os.environ["NO_PROXY"] = "*"

    from nicegui import ui, app

    from qmemory_bench.ui.i18n import t, set_lang, get_lang
    from qmemory_bench.dataset import get_dataset_preset

    # ── State ───────────────────────────────────────────────
    state: dict[str, Any] = {
        "target_url": default_target,
        "provider": "deepseek",
        "api_key": "",
        "model": "",
        "scale": "standard",
        "dataset_preset": "public-main",
        "datasets": get_dataset_preset("public-main"),
        "running": False,
        "report": None,
        # Connection status
        "conn_status": "unknown",  # unknown | ok | fail
        "conn_version": "",
        "conn_memories": 0,
        # Compare mode
        "compare_mode": False,
        "compare_providers": [],  # [{"provider": ..., "api_key": ..., "model": ...}]
        # Compare report
        "compare_report": None,
    }

    # Progress dict — updated by runner from background thread
    progress: dict[str, Any] = {
        "stage": "idle", "pct": 0.0, "detail": "",
        "session_i": 0, "session_n": 0,
        "question_i": 0, "question_n": 0,
        "qa_log": [],  # list of Q&A entries for live display
    }

    # Track the current active client id so timers skip stale pages
    _active_client: dict[str, Any] = {"id": None}

    # ── Provider list ───────────────────────────────────────
    from qmemory_bench.providers import list_providers
    providers = list_providers()
    provider_options = {p["key"]: f"{p['name']} — {p['description']}" for p in providers}

    # ── Health check ────────────────────────────────────────
    async def _check_health():
        url = state["target_url"].rstrip("/")
        try:
            async with httpx.AsyncClient(timeout=5.0) as c:
                r = await c.get(f"{url}/v1/health/")
                data = r.json()
                state["conn_status"] = "ok"
                state["conn_version"] = data.get("version", "?")
                state["conn_memories"] = data.get("memory_count", 0)
        except Exception:
            state["conn_status"] = "fail"
            state["conn_version"] = ""
            state["conn_memories"] = 0

    # ── Main page ───────────────────────────────────────────
    @ui.page("/")
    async def index():
        from nicegui import context as _nic_ctx
        _page_client_id = id(_nic_ctx.client)
        _active_client["id"] = _page_client_id

        def _client_alive() -> bool:
            """Check whether this page build is still the active one."""
            return _active_client["id"] == _page_client_id

        lang = get_lang()
        dark = ui.dark_mode()
        dark.disable()  # default light

        # ── Container ──
        with ui.column().classes("w-full max-w-4xl mx-auto p-6"):

            # ══════ Header ══════
            with ui.row().classes("w-full items-center justify-between mb-1"):
                ui.label(t("app_title")).classes("text-3xl font-bold")
                with ui.row().classes("items-center gap-3"):
                    # Eval results page link
                    ui.button(t("eval_nav"), icon="assessment",
                              on_click=lambda: ui.navigate.to("/eval")).props("flat dense size=sm color=purple")
                    # Human calibration page link
                    ui.button("🧑‍⚖️ 校准", icon="balance",
                              on_click=lambda: ui.navigate.to("/calibration")).props("flat dense size=sm color=teal")
                    # Language toggle
                    lang_btn = ui.button(t("lang_label"), icon="translate").props(
                        "flat dense size=sm"
                    )

                    def switch_lang():
                        new_lang = "en" if get_lang() == "zh" else "zh"
                        set_lang(new_lang)
                        ui.navigate.to(f"/")

                    lang_btn.on("click", switch_lang)

                    # Theme toggle
                    theme_icon = ui.icon("light_mode").classes("text-xl cursor-pointer")
                    theme_label = ui.label(t("light")).classes("text-sm cursor-pointer")

                    def toggle_theme():
                        if dark.value:
                            dark.disable()
                            theme_icon.props("name=light_mode")
                            theme_label.text = t("light")
                        else:
                            dark.enable()
                            theme_icon.props("name=dark_mode")
                            theme_label.text = t("dark")

                    theme_icon.on("click", lambda: toggle_theme())
                    theme_label.on("click", lambda: toggle_theme())

            ui.label(t("app_subtitle")).classes("text-gray-500 mb-4")

            # ══════ Methodology / Trust Panel ══════
            with ui.expansion(t("method_title"), icon="verified").classes(
                "w-full mb-4 bg-blue-50 rounded-lg"
            ).props("dense header-class='text-blue-800 font-semibold'"):
                # Overview
                ui.label(t("method_overview")).classes("text-sm mb-3")

                # 1. Cross-validation scoring
                with ui.card().classes("w-full mb-2 p-3"):
                    ui.label(f"① {t('method_cross_title')}").classes(
                        "text-sm font-bold text-blue-700"
                    )
                    ui.label(t("method_cross_body")).classes("text-xs text-gray-700")

                # 2. Noise injection
                with ui.card().classes("w-full mb-2 p-3"):
                    ui.label(f"② {t('method_noise_title')}").classes(
                        "text-sm font-bold text-blue-700"
                    )
                    ui.label(t("method_noise_body")).classes("text-xs text-gray-700")

                # 3. Multi-session & temporal
                with ui.card().classes("w-full mb-2 p-3"):
                    ui.label(f"③ {t('method_temporal_title')}").classes(
                        "text-sm font-bold text-blue-700"
                    )
                    ui.label(t("method_temporal_body")).classes("text-xs text-gray-700")

                # 4. Reproducibility
                with ui.card().classes("w-full mb-2 p-3"):
                    ui.label(f"④ {t('method_repro_title')}").classes(
                        "text-sm font-bold text-blue-700"
                    )
                    ui.label(t("method_repro_body")).classes("text-xs text-gray-700")

                # 5. Scoring methodology
                with ui.card().classes("w-full mb-2 p-3"):
                    ui.label(f"⑤ {t('method_scoring_title')}").classes(
                        "text-sm font-bold text-blue-700"
                    )
                    ui.label(t("method_scoring_body")).classes("text-xs text-gray-700")

                # 6. Dataset provenance
                with ui.card().classes("w-full mb-2 p-3"):
                    ui.label(f"⑥ {t('method_dataset_title')}").classes(
                        "text-sm font-bold text-blue-700"
                    )
                    ui.label(t("method_dataset_body")).classes("text-xs text-gray-700")

                # Trust statement
                ui.label(t("method_trust")).classes(
                    "text-sm font-semibold text-green-700 mt-2"
                )

            # ══════ QMemory Server + Connection Indicator ══════
            with ui.card().classes("w-full mb-4"):
                with ui.row().classes("w-full items-center justify-between"):
                    with ui.row().classes("items-center gap-2"):
                        conn_dot = ui.icon("circle").classes("text-gray-400 text-xs")
                        conn_label = ui.label(t("checking")).classes("text-sm text-gray-400")
                    with ui.row().classes("items-center gap-2"):
                        reset_btn = ui.button(
                            t("reset_memory"), icon="delete_forever", color="red"
                        ).props("flat dense size=sm")

                ui.label(t("server_address")).classes("text-lg font-semibold mt-2")
                target_input = ui.input(
                    "URL", value=state["target_url"],
                ).classes("w-full")
                target_input.on_value_change(lambda e: state.update(target_url=e.value))

                # Connection status update
                def _refresh_conn_ui():
                    s = state["conn_status"]
                    if s == "ok":
                        conn_dot.props("name=circle color=green")
                        conn_dot.classes(replace="text-green-500 text-xs")
                        ver = state["conn_version"]
                        mem = state["conn_memories"]
                        conn_label.text = f"{t('connected')} · v{ver} · {mem} {t('memories_count')}"
                        conn_label.classes(replace="text-sm text-green-600")
                    elif s == "fail":
                        conn_dot.props("name=circle color=red")
                        conn_dot.classes(replace="text-red-500 text-xs")
                        conn_label.text = t("disconnected")
                        conn_label.classes(replace="text-sm text-red-500")
                    else:
                        conn_dot.props("name=circle color=grey")
                        conn_dot.classes(replace="text-gray-400 text-xs")
                        conn_label.text = t("checking")
                        conn_label.classes(replace="text-sm text-gray-400")

                # Poll health every 5 seconds
                async def _health_tick():
                    if not _client_alive():
                        return  # Stale page — skip
                    if state["running"]:
                        # During benchmark: just refresh UI with cached status (don't make HTTP calls)
                        try:
                            _refresh_conn_ui()
                        except Exception:
                            pass
                        return
                    try:
                        await _check_health()
                        _refresh_conn_ui()
                    except Exception:
                        pass  # Client/element deleted; silently skip

                ui.timer(5.0, _health_tick)
                # Also check immediately
                ui.timer(0.1, _health_tick, once=True)

                # Reset
                async def do_reset_memory():
                    url = state["target_url"].rstrip("/")
                    try:
                        async with httpx.AsyncClient(timeout=10.0) as c:
                            resp = await c.request(
                                "DELETE", f"{url}/v1/memories/",
                                params={"confirm": "true"},
                            )
                            if resp.status_code in (200, 204):
                                data = resp.json()
                                count = data.get("memories_deleted", 0)
                                ui.notify(f"{t('reset_success')} ({count} 条)", type="positive")
                                # Immediately refresh connection status
                                await _check_health()
                                _refresh_conn_ui()
                            else:
                                ui.notify(f"{t('reset_fail')}: HTTP {resp.status_code}", type="negative")
                    except Exception as ex:
                        ui.notify(f"{t('reset_fail')}: {ex}", type="negative")

                async def confirm_reset():
                    with ui.dialog() as dlg, ui.card():
                        ui.label(t("reset_confirm_title")).classes("text-lg font-semibold")
                        ui.label(t("reset_confirm_msg")).classes("text-red-500")
                        with ui.row().classes("w-full justify-end gap-2 mt-4"):
                            ui.button(t("cancel"), on_click=dlg.close).props("flat")

                            async def _do():
                                dlg.close()
                                await do_reset_memory()

                            ui.button(t("confirm_reset"), color="red", on_click=_do).props("flat")
                    dlg.open()

                reset_btn.on("click", confirm_reset)

            # ══════ LLM Config — Base ══════
            with ui.card().classes("w-full mb-4"):
                with ui.row().classes("w-full items-center justify-between"):
                    ui.label(t("llm_config")).classes("text-lg font-semibold")
                    compare_switch = ui.switch(t("compare_mode"), value=state["compare_mode"])

                # When compare mode is OFF this is "the LLM"; when ON it
                # becomes the "base LLM" that others are compared against.
                base_tag = ui.label(t("base_llm_tag")).classes(
                    "text-xs font-semibold px-2 py-0.5 rounded bg-blue-100 text-blue-700 mb-2"
                )
                base_tag.visible = state["compare_mode"]

                # Primary provider (always visible)
                provider_select = ui.select(
                    provider_options, value=state["provider"], label="Provider",
                ).classes("w-full mb-2")
                provider_select.on_value_change(lambda e: state.update(provider=e.value))

                api_key_input = ui.input(
                    t("api_key"), value=state["api_key"],
                    password=True, password_toggle_button=True,
                ).classes("w-full mb-2")
                api_key_input.on_value_change(lambda e: state.update(api_key=e.value))

                model_input = ui.input(
                    t("model_hint"), value=state["model"],
                ).classes("w-full")
                model_input.on_value_change(lambda e: state.update(model=e.value))

            # ══════ Compare LLM section (separate card, prominent) ══════
            compare_card = ui.card().classes(
                "w-full mb-4 border-2 border-dashed border-orange-300 bg-orange-50"
            )
            compare_card.visible = state["compare_mode"]

            with compare_card:
                with ui.row().classes("w-full items-center gap-2 mb-1"):
                    ui.icon("compare_arrows").classes("text-orange-500 text-xl")
                    ui.label(t("compare_section_title")).classes(
                        "text-lg font-semibold text-orange-700"
                    )
                ui.label(t("compare_hint")).classes("text-xs text-gray-500 mb-3")

                # Container that holds dynamically-added comparison rows
                compare_container = ui.column().classes("w-full gap-2")

                def _add_compare_row():
                    entry = {"provider": "zhipu", "api_key": "", "model": ""}
                    state["compare_providers"].append(entry)
                    idx = len(state["compare_providers"])

                    with compare_container:
                        with ui.card().classes(
                            "w-full p-3 border border-orange-200"
                        ) as row_card:
                            with ui.row().classes("w-full items-center justify-between mb-2"):
                                ui.label(f"{t('provider_label')} {idx}").classes(
                                    "text-sm font-bold text-orange-600"
                                )

                                def _remove(rc=row_card, en=entry):
                                    if en in state["compare_providers"]:
                                        state["compare_providers"].remove(en)
                                    rc.delete()

                                ui.button(icon="close", color="red", on_click=_remove).props(
                                    "flat dense round size=sm"
                                )

                            sel = ui.select(
                                provider_options, value=entry["provider"], label="Provider",
                            ).classes("w-full mb-1")
                            sel.on_value_change(lambda e, en=entry: en.update(provider=e.value))

                            key_inp = ui.input(
                                "API Key", password=True, password_toggle_button=True,
                            ).classes("w-full mb-1")
                            key_inp.on_value_change(lambda e, en=entry: en.update(api_key=e.value))

                            mod_inp = ui.input(
                                t("model_hint"),
                            ).classes("w-full")
                            mod_inp.on_value_change(lambda e, en=entry: en.update(model=e.value))

                # Full-width prominent "add" button
                add_compare_btn = ui.button(
                    t("add_provider"), icon="add_circle",
                    on_click=_add_compare_row,
                    color="orange",
                ).props("outline").classes("w-full mt-2 text-base")

            def _toggle_compare(e):
                on = e.value
                state["compare_mode"] = on
                compare_card.visible = on
                base_tag.visible = on
                # Auto-add one comparison row when opening for first time
                if on and not state["compare_providers"]:
                    _add_compare_row()

            compare_switch.on_value_change(_toggle_compare)

            ui.label(t("embedding_note")).classes("text-gray-500 text-sm mb-4")

            # ══════ Datasets ══════
            with ui.card().classes("w-full mb-4"):
                ui.label(t("eval_scope")).classes("text-lg font-semibold mb-2")
                from qmemory_bench.dataset import (
                    DATASET_AUTHORITY_LABELS,
                    DATASET_GROUP_LABELS,
                    DATASET_PRESET_LABELS,
                    get_dataset_preset,
                    get_grouped_datasets,
                    infer_dataset_preset,
                )
                ds_checks: dict[str, Any] = {}
                preset_badge = ui.label("").classes("text-sm text-blue-700 mb-2")

                def _refresh_preset_badge() -> None:
                    preset_name = state.get("dataset_preset", "custom")
                    if preset_name == "custom":
                        preset_badge.text = f"{t('current_selection')}: {t('preset_custom')}"
                    else:
                        preset_badge.text = (
                            f"{t('current_selection')}: "
                            f"{DATASET_PRESET_LABELS.get(preset_name, preset_name)}"
                        )

                def _apply_dataset_selection(names: list[str], preset_name: str) -> None:
                    selected = list(dict.fromkeys(names))
                    state["datasets"] = selected
                    state["dataset_preset"] = preset_name
                    for name, cb in ds_checks.items():
                        cb.value = name in selected
                    _refresh_preset_badge()

                def _sync_ds_state():
                    """Keep state['datasets'] in sync with checkbox values."""
                    state["datasets"] = [
                        name for name, cb in ds_checks.items() if cb.value
                    ]
                    state["dataset_preset"] = infer_dataset_preset(state["datasets"])
                    _refresh_preset_badge()

                with ui.row().classes("w-full gap-2 mb-3 flex-wrap"):
                    ui.button(
                        DATASET_PRESET_LABELS["public-main"],
                        on_click=lambda: _apply_dataset_selection(
                            get_dataset_preset("public-main"), "public-main"
                        ),
                    ).props("outline size=sm color=primary")
                    ui.button(
                        DATASET_PRESET_LABELS["release-full"],
                        on_click=lambda: _apply_dataset_selection(
                            get_dataset_preset("release-full"), "release-full"
                        ),
                    ).props("outline size=sm color=primary")
                    ui.button(
                        DATASET_PRESET_LABELS["supporting"],
                        on_click=lambda: _apply_dataset_selection(
                            get_dataset_preset("supporting"), "supporting"
                        ),
                    ).props("outline size=sm")
                    ui.button(
                        DATASET_PRESET_LABELS["regression"],
                        on_click=lambda: _apply_dataset_selection(
                            get_dataset_preset("regression"), "regression"
                        ),
                    ).props("outline size=sm")
                    ui.button(
                        DATASET_PRESET_LABELS["all"],
                        on_click=lambda: _apply_dataset_selection(
                            get_dataset_preset("all"), "all"
                        ),
                    ).props("outline size=sm")

                ui.label(t("preset_help")).classes("text-xs text-gray-500 mb-2")
                _refresh_preset_badge()

                for group_name, items in get_grouped_datasets().items():
                    with ui.card().classes("w-full mb-2 p-3 bg-gray-50"):
                        ui.label(DATASET_GROUP_LABELS.get(group_name, group_name)).classes(
                            "text-sm font-semibold mb-2"
                        )
                        for ds_name, ds_info in items:
                            checked = ds_name in state["datasets"]
                            src = ds_info.get("source", "custom")
                            src_note = ds_info.get("source_note", "")
                            authority = DATASET_AUTHORITY_LABELS.get(
                                ds_info.get("authority", "custom"),
                                ds_info.get("authority", "custom"),
                            )
                            with ui.row().classes("w-full items-start"):
                                cb = ui.checkbox(
                                    f"{ds_name} — {ds_info['description']}",
                                    value=checked,
                                    on_change=lambda _e: _sync_ds_state(),
                                )
                                ds_checks[ds_name] = cb
                            ui.label(
                                f"    [{authority} / {src}] {src_note}"
                            ).classes("text-xs text-gray-400 ml-8 -mt-1 mb-1")

            # ══════ Scale ══════
            with ui.card().classes("w-full mb-4"):
                ui.label(t("eval_scale")).classes("text-lg font-semibold mb-2")
                scale_radio = ui.radio(
                    {
                        "quick": t("scale_quick"),
                        "standard": t("scale_standard"),
                        "full": t("scale_full"),
                    },
                    value=state["scale"],
                ).props("inline")
                scale_radio.on_value_change(lambda e: state.update(scale=e.value))

            # ══════ Run Button ══════
            run_label = t("start_compare") if state["compare_mode"] else t("start_eval")
            run_btn = ui.button(run_label, color="green").classes(
                "w-full text-xl py-4 mb-2"
            )

            # Progress area
            with ui.card().classes("w-full mb-4") as progress_card:
                progress_card.visible = state["running"]
                progress_label = ui.label("").classes("text-sm font-medium mb-1")
                progress_bar = ui.linear_progress(value=0, show_value=False).classes("w-full mb-1")
                progress_detail = ui.label("").classes("text-xs text-gray-400")

            # Disable run button if benchmark is already in progress (page refresh)
            if state["running"]:
                run_btn.disable()

            # ══════ Q&A Live Log ══════
            with ui.card().classes("w-full mb-4") as qa_card:
                qa_card.visible = bool(state["running"] or progress.get("qa_log"))
                with ui.row().classes("w-full items-center justify-between mb-2"):
                    ui.label(t("qa_log_title")).classes("text-lg font-semibold")
                    qa_count_label = ui.label("").classes("text-sm text-gray-400")
                qa_container = ui.column().classes(
                    "w-full gap-2"
                ).style("max-height: 420px; overflow-y: auto; padding-right: 8px")

            # Track how many Q&A entries have been rendered into qa_container
            qa_rendered = {"n": 0}

            def _render_qa_entry(entry):
                """Render one Q&A card inside qa_container."""
                score = entry["score"]
                clr = "green" if score >= 7 else "orange" if score >= 4 else "red"
                mem_count = entry.get("memory_count", 0)
                prec = entry.get("precision", 1.0)
                is_adv = entry.get("is_adversarial", False)
                with ui.card().classes(
                    f"w-full p-3 border-l-4 border-{clr}-400 bg-gray-50"
                ).style("min-width: 0"):
                    # Header: dataset/category, memory count, score badge
                    with ui.row().classes("w-full items-center justify-between"):
                        cat_label = entry['category']
                        if is_adv:
                            cat_label = f"⚔️ {cat_label}"
                        ui.label(
                            f"[{entry['dataset']}/{cat_label}] {entry['id']}"
                        ).classes("text-xs text-gray-500")
                        with ui.row().classes("items-center gap-2"):
                            mem_clr = "green" if mem_count > 0 else "red"
                            ui.badge(
                                f"{mem_count} {t('memories_count')}",
                                color=mem_clr,
                            ).props("outline rounded")
                            # Precision badge
                            prec_clr = "green" if prec >= 0.8 else "orange" if prec >= 0.5 else "red"
                            ui.badge(f"P={prec:.0%}", color=prec_clr).props("outline rounded")
                            ui.badge(f"{score}/10", color=clr).props("rounded")

                    # Question
                    ui.label(f"❓ {entry['query']}").classes(
                        "text-sm font-medium mt-1"
                    )
                    # Expected answer
                    ui.label(
                        f"✅ {t('qa_expected')}: {entry['expected'][:200]}"
                    ).classes("text-xs text-green-700")

                    # Recalled memories summary
                    mem_texts = entry.get("memory_texts", [])
                    recall_ctx = entry.get("recall", "")
                    if mem_texts:
                        with ui.column().classes("ml-2 mt-1 gap-0"):
                            for mt in mem_texts[:3]:
                                ui.label(f"📌 {mt}").classes(
                                    "text-xs text-blue-600"
                                )
                            if len(mem_texts) > 3:
                                ui.label(
                                    f"  ...+{len(mem_texts) - 3} more"
                                ).classes("text-xs text-gray-400")
                    elif recall_ctx:
                        ui.label(
                            f"🔍 {t('qa_recalled')}: {recall_ctx[:300]}"
                        ).classes("text-xs text-gray-500")
                    else:
                        ui.label(
                            f"❌ {t('qa_recalled')}: (无召回结果)"
                        ).classes("text-xs text-red-400")

                    # Judge reason
                    if entry.get("reason"):
                        ui.label(
                            f"💬 {entry['reason'][:400]}"
                        ).classes("text-xs text-gray-400 italic mt-1")

            # On page refresh, restore already-collected Q&A entries
            existing_qa = progress.get("qa_log", [])
            if existing_qa:
                for entry in existing_qa:
                    with qa_container:
                        _render_qa_entry(entry)
                qa_rendered["n"] = len(existing_qa)
                qa_count_label.text = f"{len(existing_qa)} {t('qa_items')}"

            # Timer to update progress UI
            def _progress_tick():
                if not _client_alive():
                    return  # This page instance is stale; skip UI updates
                if not state["running"]:
                    return

                # Part 1: Always update progress text (lightweight, must not fail)
                try:
                    pct = progress.get("pct", 0)
                    detail = progress.get("detail", "")
                    stage = progress.get("stage", "")
                    progress_bar.value = pct
                    progress_detail.text = detail
                    if stage == "error":
                        progress_label.text = f"❌ {detail}"
                    elif stage == "done":
                        progress_label.text = f"✅ {detail}"
                    elif stage in ("injecting",):
                        si = progress.get("session_i", 0)
                        sn = progress.get("session_n", 0)
                        progress_label.text = f"{t('prog_injecting')} {si}/{sn} (LLM提取中…)"
                    elif stage == "injected":
                        progress_label.text = f"✅ {detail}"
                    elif stage in ("evaluating",):
                        qi = progress.get("question_i", 0)
                        qn = progress.get("question_n", 0)
                        progress_label.text = f"{t('prog_evaluating')} {qi}/{qn}"
                    elif stage == "cleanup":
                        progress_label.text = t("prog_cleanup")
                    elif stage == "connecting":
                        progress_label.text = t("prog_connecting")
                    elif stage == "compare":
                        progress_label.text = detail
                    else:
                        progress_label.text = detail
                except Exception as exc:
                    logger.warning("Progress text update failed: %s", exc)

                # Part 2: Incrementally render Q&A entries (separate try/except)
                try:
                    qa_log = progress.get("qa_log", [])
                    new_count = len(qa_log)
                    if new_count > qa_rendered["n"]:
                        qa_card.visible = True
                        new_entries = qa_log[qa_rendered["n"]:]
                        for entry in new_entries:
                            with qa_container:
                                _render_qa_entry(entry)
                        qa_rendered["n"] = new_count
                        qa_count_label.text = f"{new_count} {t('qa_items')}"
                        # Auto-scroll to bottom
                        ui.run_javascript(f'''
                            var el = getElement({qa_container.id}).$el;
                            if (el) el.scrollTop = el.scrollHeight;
                        ''')
                except Exception as exc:
                    logger.warning("Q&A log render failed: %s", exc)

            ui.timer(0.5, _progress_tick)

            # ══════ Results Area ══════
            results_container = ui.column().classes("w-full")

            # Restore results if benchmark already completed (e.g. page was refreshed)
            if not state["running"] and state.get("report"):
                _show_single_results(results_container, state["report"])
            elif not state["running"] and state.get("compare_report"):
                _show_comparison_results(results_container, state["compare_report"])

            # ── Run logic ──
            async def start_benchmark():
                if state["running"]:
                    return
                if not state["api_key"]:
                    ui.notify(t("enter_api_key"), type="warning")
                    return

                state["running"] = True
                run_btn.disable()
                progress_card.visible = True
                progress["qa_log"] = []  # Reset Q&A log
                qa_rendered["n"] = 0
                qa_container.clear()
                qa_card.visible = True
                qa_count_label.text = ""
                progress.update(stage="connecting", pct=0.02, detail=t("prog_connecting"))

                # Ensure connection status is fresh before entering running state
                try:
                    await _check_health()
                    _refresh_conn_ui()
                except Exception:
                    pass

                selected = state["datasets"][:]
                if not selected:
                    # Fallback: use checkbox state directly
                    selected = [name for name, cb in ds_checks.items() if cb.value]
                if not selected:
                    selected = ["longmemeval-s"]

                async def _do_run():
                    try:
                        if state["compare_mode"] and state["compare_providers"]:
                            await _run_comparison(selected)
                        else:
                            await _run_single(selected)
                    finally:
                        state["running"] = False
                        if _client_alive():
                            try:
                                run_btn.enable()
                            except Exception:
                                pass

                # Launch as background task with proper NiceGUI client context
                from nicegui import background_tasks
                background_tasks.create(_do_run(), name="benchmark_run")

            async def _run_single(selected: list[str]):
                from qmemory_bench.runner import BenchmarkConfig, run_benchmark

                config = BenchmarkConfig(
                    target_url=state["target_url"],
                    provider=state["provider"],
                    api_key=state["api_key"],
                    model=state["model"],
                    scale=state["scale"],
                    dataset_names=selected,
                    dataset_preset=state["dataset_preset"],
                )

                try:
                    report = await run_benchmark(config, progress)
                    state["report"] = report
                    if _client_alive():
                        _show_single_results(results_container, report)
                except Exception as ex:
                    progress.update(stage="error", detail=f"{t('prog_fail')}: {ex}")
                    if _client_alive():
                        ui.notify(f"Error: {ex}", type="negative")

            async def _run_comparison(selected: list[str]):
                from qmemory_bench.runner import ComparisonConfig, run_comparison

                all_provs = [
                    {"provider": state["provider"],
                     "api_key": state["api_key"],
                     "model": state["model"]},
                ]
                all_provs.extend(state["compare_providers"])

                # Validate all have keys
                for p in all_provs:
                    if not p.get("api_key"):
                        ui.notify(f"{p['provider']}: {t('enter_api_key')}", type="warning")
                        return

                config = ComparisonConfig(
                    target_url=state["target_url"],
                    providers=all_provs,
                    scale=state["scale"],
                    dataset_names=selected,
                    dataset_preset=state["dataset_preset"],
                )

                try:
                    report = await run_comparison(config, progress)
                    state["compare_report"] = report
                    if _client_alive():
                        _show_comparison_results(results_container, report)
                except Exception as ex:
                    progress.update(stage="error", detail=f"{t('prog_fail')}: {ex}")
                    if _client_alive():
                        ui.notify(f"Error: {ex}", type="negative")

            run_btn.on("click", start_benchmark)

            # Update run button label on compare toggle
            def _update_run_label(e):
                run_btn.text = t("start_compare") if e.value else t("start_eval")

            compare_switch.on_value_change(_update_run_label)

    # ── Single-run results ──────────────────────────────────
    def _show_single_results(container, report):
        container.clear()
        with container:
            from qmemory_bench.runner import TARGETS
            from qmemory_bench.dataset import DATASET_PRESET_LABELS

            target_met = report.overall >= 85.0
            color = "green" if target_met else "red"
            status = f"✅ {t('pass_label')}" if target_met else f"❌ {t('fail_label')}"

            ui.separator()
            ui.label(t("results_title")).classes("text-xl font-bold mt-2")

            ui.label(
                f"{t('overall')}: {report.overall:.1f}% ({t('target')}≥85%) {status}"
            ).classes(f"text-2xl font-bold text-{color}-500 mb-2")

            ui.label(
                f"{t('current_selection')}: "
                f"{DATASET_PRESET_LABELS.get(getattr(report, 'dataset_preset', 'custom'), t('preset_custom'))}"
            ).classes("text-sm text-blue-700 mb-2")

            ui.linear_progress(
                value=report.overall / 100, show_value=True,
                color="green" if target_met else "red",
            ).classes("w-full mb-4")

            # Show per-dataset precision summary
            for ds_name, ds_rpt in report.datasets.items():
                ds_prec = getattr(ds_rpt, 'overall_precision', 0.0)
                if ds_prec > 0:
                    prec_clr = "green" if ds_prec >= 0.8 else "orange" if ds_prec >= 0.5 else "red"
                    ui.label(
                        f"🎯 {ds_name} 精确率: {ds_prec:.1%}"
                    ).classes(f"text-sm text-{prec_clr}-600")

            # Info row
            ui.label(
                f"QMemory {report.qmemory_version} · {report.llm_provider}/{report.llm_model} "
                f"· {report.scale} · {t('duration')} {report.duration}s"
            ).classes("text-xs text-gray-400 mb-4")

            # Per-dataset tables
            for ds_name, ds_report in report.datasets.items():
                ui.label(f"{t('dataset_label')}: {ds_name}").classes(
                    "text-lg font-semibold mb-2"
                )
                _render_score_table(ds_report, TARGETS)

                # Concurrency metrics card (if available)
                cm = getattr(ds_report, 'concurrency', None)
                if cm and cm.total_requests > 0:
                    with ui.card().classes("w-full p-3 mb-4 bg-blue-50 border border-blue-200"):
                        ui.label(f"⚡ 并发压测 (N={cm.concurrency})").classes(
                            "text-sm font-semibold text-blue-700"
                        )
                        with ui.row().classes("gap-6 mt-1"):
                            _metric = lambda lbl, val: ui.label(f"{lbl}: {val}").classes("text-xs")
                            _metric("P50", f"{cm.p50_ms:.0f}ms")
                            _metric("P95", f"{cm.p95_ms:.0f}ms")
                            _metric("P99", f"{cm.p99_ms:.0f}ms")
                            _metric("AVG", f"{cm.avg_ms:.0f}ms")
                            _metric("MAX", f"{cm.max_ms:.0f}ms")

                        with ui.row().classes("gap-6"):
                            _metric("请求数", str(cm.total_requests))
                            err_clr = "text-red-600" if cm.errors > 0 else "text-green-600"
                            ui.label(f"错误: {cm.errors}").classes(f"text-xs {err_clr}")
                            drop_clr = "text-red-600" if cm.accuracy_drop > 5 else "text-green-600"
                            ui.label(f"精度下降: {cm.accuracy_drop:.1f}pp").classes(f"text-xs {drop_clr}")

            # Weakness analysis
            _render_weaknesses(report, TARGETS)

            # Export buttons
            _render_export_buttons(report)

    # ── Comparison results ──────────────────────────────────
    def _show_comparison_results(container, comp_report):
        container.clear()
        with container:
            from qmemory_bench.runner import TARGETS
            from qmemory_bench.dataset import DATASET_PRESET_LABELS

            ui.separator()
            ui.label(t("compare_results")).classes("text-xl font-bold mt-2")
            ui.label(
                f"{t('current_selection')}: "
                f"{DATASET_PRESET_LABELS.get(getattr(comp_report, 'dataset_preset', 'custom'), t('preset_custom'))}"
            ).classes("text-sm text-blue-700 mb-3")

            labels = list(comp_report.reports.keys())
            reports = [comp_report.reports[k] for k in labels]

            # Overall comparison row
            with ui.row().classes("w-full items-center gap-4 mb-4"):
                for lbl, rpt in zip(labels, reports):
                    met = rpt.overall >= 85.0
                    clr = "green" if met else "red"
                    with ui.card().classes("flex-1 text-center p-3"):
                        ui.label(lbl).classes("text-sm font-semibold")
                        ui.label(f"{rpt.overall:.1f}%").classes(
                            f"text-3xl font-bold text-{clr}-500"
                        )
                        ui.label(f"{rpt.duration}s").classes("text-xs text-gray-400")

            # Find overall winner
            if reports:
                best = max(reports, key=lambda r: r.overall)
                best_lbl = labels[reports.index(best)]
                ui.label(f"🏆 {t('winner')}: {best_lbl} ({best.overall:.1f}%)").classes(
                    "text-lg font-semibold text-green-600 mb-4"
                )

            # Detailed comparison table
            # Collect all categories across all reports
            all_cats: dict[str, dict[str, float]] = {}
            for lbl, rpt in zip(labels, reports):
                for ds_report in rpt.datasets.values():
                    for cat, info in ds_report.categories.items():
                        if cat not in all_cats:
                            all_cats[cat] = {}
                        all_cats[cat][lbl] = info["score"]

            columns = [
                {"name": "category", "label": t("category"),
                 "field": "category", "align": "left"},
                {"name": "target", "label": t("target"),
                 "field": "target", "align": "right"},
            ]
            for lbl in labels:
                columns.append(
                    {"name": lbl, "label": lbl, "field": lbl, "align": "right"}
                )
            if len(labels) == 2:
                columns.append(
                    {"name": "diff", "label": t("diff"), "field": "diff", "align": "right"}
                )

            rows = []
            for cat, scores in all_cats.items():
                tgt = TARGETS.get(cat, 80.0)
                row: dict[str, Any] = {
                    "category": cat,
                    "target": f"{tgt:.0f}%",
                }
                for lbl in labels:
                    s = scores.get(lbl, 0)
                    row[lbl] = f"{s:.1f}%"
                if len(labels) == 2:
                    s1 = scores.get(labels[0], 0)
                    s2 = scores.get(labels[1], 0)
                    d = s1 - s2
                    row["diff"] = f"{d:+.1f}"
                rows.append(row)

            ui.table(columns=columns, rows=rows).classes("w-full mb-4")

            # Export: download all reports as one JSON
            _render_comparison_export(comp_report)

    # ── Shared rendering helpers ────────────────────────────
    def _render_score_table(ds_report, targets):
        columns = [
            {"name": "category", "label": t("category"),
             "field": "category", "align": "left"},
            {"name": "score", "label": t("score"),
             "field": "score", "align": "right"},
            {"name": "precision", "label": "精确率",
             "field": "precision", "align": "right"},
            {"name": "target", "label": t("target"),
             "field": "target", "align": "right"},
            {"name": "status", "label": t("status"),
             "field": "status", "align": "center"},
            {"name": "count", "label": t("count"),
             "field": "count", "align": "right"},
        ]
        rows = []
        for cat, info in ds_report.categories.items():
            tgt = targets.get(cat, 80.0)
            met = info["score"] >= tgt
            prec = info.get("precision", 0.0)
            rows.append({
                "category": cat,
                "score": f'{info["score"]:.1f}%',
                "precision": f'{prec:.1%}',
                "target": f"{tgt:.0f}%",
                "status": "✓" if met else "✗",
                "count": info["count"],
            })
        ui.table(columns=columns, rows=rows).classes("w-full mb-4")

    def _render_weaknesses(report, targets):
        weaknesses = []
        for ds_report in report.datasets.values():
            for cat, info in ds_report.categories.items():
                tgt = targets.get(cat, 80.0)
                if info["score"] < tgt:
                    weaknesses.append((cat, info["score"], tgt))

        if weaknesses:
            ui.label(f"⚠️ {t('weakness_title')}").classes(
                "text-lg font-semibold text-yellow-600 mt-2"
            )
            for cat, score, tgt in weaknesses:
                ui.label(
                    f"  • {cat}: {score:.1f}% — {t('below_target')} ({tgt:.0f}%)"
                ).classes("text-yellow-600 text-sm")

    def _render_export_buttons(report):
        from qmemory_bench.runner import report_to_dict

        with ui.row().classes("w-full gap-3 mt-4"):
            # Download JSON
            async def _download_json():
                data = report_to_dict(report)
                text = json.dumps(data, indent=2, ensure_ascii=False)
                tmp = Path(tempfile.mktemp(suffix=".json", prefix="qmemory_bench_"))
                tmp.write_text(text, encoding="utf-8")
                ui.download(str(tmp), filename="qmemory_benchmark_report.json")
                ui.notify(t("download_ready"), type="positive")

            ui.button(t("download_json"), icon="download",
                      on_click=_download_json).props("outline")

            # Copy text report
            async def _copy_text():
                lines = [
                    f"QMemory Benchmark Report",
                    f"========================",
                    f"Overall: {report.overall:.1f}% (target ≥85%)",
                    f"Preset: {getattr(report, 'dataset_preset', 'custom')}",
                    f"QMemory: {report.qmemory_version}",
                    f"LLM: {report.llm_provider}/{report.llm_model}",
                    f"Scale: {report.scale}  Duration: {report.duration}s",
                    f"Timestamp: {report.timestamp}",
                    "",
                ]
                from qmemory_bench.runner import TARGETS
                for ds_name, ds_rpt in report.datasets.items():
                    lines.append(f"Dataset: {ds_name}")
                    for cat, info in ds_rpt.categories.items():
                        tgt = TARGETS.get(cat, 80.0)
                        st = "PASS" if info["score"] >= tgt else "FAIL"
                        lines.append(
                            f"  {cat:30s} {info['score']:6.1f}%  "
                            f"target={tgt:.0f}%  {st}"
                        )
                    lines.append("")
                text = "\n".join(lines)
                ui.run_javascript(
                    f"navigator.clipboard.writeText({json.dumps(text)})"
                )
                ui.notify(t("copied"), type="positive")

            ui.button(t("copy_text"), icon="content_copy",
                      on_click=_copy_text).props("outline")

    def _render_comparison_export(comp_report):
        from qmemory_bench.runner import report_to_dict

        with ui.row().classes("w-full gap-3 mt-4"):
            async def _download_compare():
                data = {
                    "comparison": True,
                    "timestamp": comp_report.timestamp,
                    "scale": comp_report.scale,
                    "reports": {
                        k: report_to_dict(v) for k, v in comp_report.reports.items()
                    },
                }
                text = json.dumps(data, indent=2, ensure_ascii=False)
                tmp = Path(tempfile.mktemp(suffix=".json", prefix="qmemory_compare_"))
                tmp.write_text(text, encoding="utf-8")
                ui.download(str(tmp), filename="qmemory_comparison_report.json")
                ui.notify(t("download_ready"), type="positive")

            ui.button(t("download_json"), icon="download",
                      on_click=_download_compare).props("outline")

    # ── Eval page: LoCoMo flat vs hierarchy ────────────────
    def _load_eval_results() -> dict | None:
        """Load the latest eval results JSON."""
        results_dir = Path(__file__).resolve().parent.parent.parent.parent / "results"
        # Try rescore first, then any locomo_eval file
        for pattern in ["locomo_eval_rescore.json", "locomo_eval_*.json"]:
            files = sorted(results_dir.glob(pattern))
            if files:
                with open(files[-1], "r", encoding="utf-8") as f:
                    return json.load(f)
        return None

    @ui.page("/eval")
    async def eval_page():
        from qmemory_bench.ui.i18n import t

        dark = ui.dark_mode()
        dark.disable()

        with ui.column().classes("w-full max-w-5xl mx-auto p-6"):

            # Header
            with ui.row().classes("w-full items-center justify-between mb-2"):
                ui.label(t("eval_title")).classes("text-3xl font-bold")
                with ui.row().classes("gap-2"):
                    ui.button("🧑‍⚖️ 人工校准", icon="balance",
                              on_click=lambda: ui.navigate.to("/calibration")).props("flat dense color=teal")
                    ui.button("← " + t("app_title"), icon="arrow_back",
                              on_click=lambda: ui.navigate.to("/")).props("flat dense")
            ui.label(t("eval_subtitle")).classes("text-gray-500 mb-4")

            data = _load_eval_results()
            if not data:
                ui.label(t("eval_no_results")).classes("text-red-500 text-xl")
                return

            flat_overall = data.get("flat_overall", 0)
            hier_overall = data.get("hier_overall", 0)
            delta = data.get("delta", 0)
            flat_cats = data.get("flat_by_category", {})
            hier_cats = data.get("hier_by_category", {})

            # ── Overall Score Cards ──
            with ui.row().classes("w-full gap-4 mb-4"):
                with ui.card().classes("flex-1 text-center p-4"):
                    ui.label(t("eval_flat")).classes("text-sm font-semibold text-gray-500")
                    clr_f = "green" if flat_overall >= 70 else "orange" if flat_overall >= 50 else "red"
                    ui.label(f"{flat_overall:.1f}%").classes(f"text-4xl font-bold text-{clr_f}-500")
                with ui.card().classes("flex-1 text-center p-4 border-2 border-purple-300"):
                    ui.label(t("eval_hier")).classes("text-sm font-semibold text-purple-600")
                    clr_h = "green" if hier_overall >= 70 else "orange" if hier_overall >= 50 else "red"
                    ui.label(f"{hier_overall:.1f}%").classes(f"text-4xl font-bold text-{clr_h}-500")
                with ui.card().classes("flex-1 text-center p-4 bg-green-50"):
                    ui.label(t("eval_delta")).classes("text-sm font-semibold text-green-600")
                    ui.label(f"+{delta:.1f}pp").classes("text-4xl font-bold text-green-600")

            # ── Category Comparison Table ──
            ui.label(t("eval_summary")).classes("text-xl font-semibold mb-2")
            cat_order = ["recall-accuracy", "multi-hop", "temporal", "logical-reasoning", "noise-resist"]
            columns = [
                {"name": "cat", "label": t("eval_cat"), "field": "cat", "align": "left"},
                {"name": "flat", "label": t("eval_flat"), "field": "flat", "align": "right"},
                {"name": "hier", "label": t("eval_hier"), "field": "hier", "align": "right"},
                {"name": "delta", "label": t("eval_delta"), "field": "delta", "align": "right"},
            ]
            rows = []
            for cat in cat_order:
                f = flat_cats.get(cat, 0)
                h = hier_cats.get(cat, 0)
                d = h - f
                rows.append({
                    "cat": cat,
                    "flat": f"{f:.1f}%",
                    "hier": f"{h:.1f}%",
                    "delta": f"{d:+.1f}pp",
                })
            ui.table(columns=columns, rows=rows).classes("w-full mb-6")

            # ── Per-question Detail ──
            questions = data.get("questions", [])
            if questions:
                ui.label(t("eval_detail")).classes("text-xl font-semibold mb-2")

                # Category filter
                all_cats = sorted(set(q.get("category", "") for q in questions))
                cat_filter = ui.select(
                    ["all"] + all_cats, value="all", label=t("eval_cat")
                ).classes("w-48 mb-2")

                q_container = ui.column().classes("w-full gap-2")

                def _render_questions(filter_cat: str = "all"):
                    q_container.clear()
                    with q_container:
                        for q in questions:
                            qcat = q.get("category", "")
                            if filter_cat != "all" and qcat != filter_cat:
                                continue
                            sf = q.get("score_flat", 0)
                            sh = q.get("score_hier", 0)
                            d = sh - sf
                            # Color: green if hier better, red if worse, gray if same
                            border_clr = "green" if d > 0 else "red" if d < 0 else "gray"
                            with ui.card().classes(
                                f"w-full p-3 border-l-4 border-{border_clr}-400"
                            ):
                                with ui.row().classes("w-full items-center justify-between"):
                                    with ui.row().classes("items-center gap-2"):
                                        ui.badge(qcat, color="blue").props("outline rounded")
                                        ui.badge(q.get("difficulty", ""), color="grey").props("outline rounded")
                                    with ui.row().classes("items-center gap-2"):
                                        f_clr = "green" if sf >= 7 else "orange" if sf >= 4 else "red"
                                        h_clr = "green" if sh >= 7 else "orange" if sh >= 4 else "red"
                                        ui.badge(f"Flat:{sf}", color=f_clr).props("rounded")
                                        ui.badge(f"Hier:{sh}", color=h_clr).props("rounded")
                                        d_clr = "green" if d > 0 else "red" if d < 0 else "grey"
                                        ui.badge(f"{d:+d}", color=d_clr).props("rounded")

                                ui.label(f"❓ {q.get('query', '')}").classes("text-sm font-medium mt-1")
                                ui.label(f"✅ {q.get('expected', '')[:200]}").classes("text-xs text-green-700")

                                # Expand for reasons and context
                                with ui.expansion("详情 / Details").props("dense"):
                                    if q.get("reason_flat"):
                                        ui.label(f"Flat 理由: {q['reason_flat']}").classes("text-xs text-gray-500")
                                    if q.get("reason_hier"):
                                        ui.label(f"Hier 理由: {q['reason_hier']}").classes("text-xs text-purple-500")

                _render_questions()
                cat_filter.on_value_change(lambda e: _render_questions(e.value))

            # ── Interactive Search Test ──
            ui.separator().classes("my-4")
            ui.label(t("eval_search")).classes("text-xl font-semibold mb-1")
            ui.label(t("eval_search_subtitle")).classes("text-gray-500 text-sm mb-3")

            search_input = ui.input("Query", placeholder="输入搜索词...").classes("w-full mb-2")
            search_btn = ui.button(t("eval_search_btn"), icon="search", color="purple").classes("mb-3")

            search_results = ui.column().classes("w-full gap-3")

            async def do_search():
                query = search_input.value
                if not query:
                    return
                search_results.clear()
                url = state["target_url"].rstrip("/")

                async with httpx.AsyncClient(timeout=120.0) as c:
                    results = {}
                    for user_id, label, hierarchy in [
                        ("eval_locomo_flat", t("eval_flat"), "false"),
                        ("eval_locomo_hier", t("eval_hier"), "true"),
                    ]:
                        try:
                            r = await c.get(
                                f"{url}/v1/memories/search/",
                                params={"q": query, "user_id": user_id, "limit": 10, "hierarchy": hierarchy},
                            )
                            data = r.json()
                            results[label] = {
                                "memories": data.get("memories", []),
                                "context": data.get("context", "")
                            }
                        except Exception as ex:
                            results[label] = {"memories": [], "context": f"Error: {ex}"}

                with search_results:
                    with ui.row().classes("w-full gap-4"):
                        for label, res in results.items():
                            with ui.card().classes("flex-1 p-3"):
                                mems = res["memories"]
                                clr = "purple" if "L2" in label or "Hier" in label or "层级" in label else "blue"
                                ui.label(f"{label} ({len(mems)} memories)").classes(f"text-lg font-semibold text-{clr}-600 mb-2")
                                for m in mems[:5]:
                                    ui.label(f"📌 {m.get('text', '')[:150]}").classes("text-xs text-gray-700 mb-1")
                                if len(mems) > 5:
                                    ui.label(f"...+{len(mems)-5} more").classes("text-xs text-gray-400")
                                with ui.expansion(t("eval_recalled_ctx")).props("dense").classes("mt-2"):
                                    ui.label(res["context"][:1000] if res["context"] else "(empty)").classes("text-xs text-gray-500 whitespace-pre-wrap")

            search_btn.on("click", do_search)

    # ── Calibration page: 人工校准 ─────────────────────────
    def _load_comprehensive_results() -> dict | None:
        """Load the latest comprehensive eval result."""
        results_dir = Path(__file__).resolve().parent.parent.parent.parent / "results"
        for pattern in ["comprehensive_*.json"]:
            files = sorted(results_dir.glob(pattern))
            if files:
                with open(files[-1], "r", encoding="utf-8") as f:
                    return json.load(f)
        # Fallback to locomo eval
        return _load_eval_results()

    def _load_calibration_file() -> dict | None:
        """Load existing calibration file if any."""
        results_dir = Path(__file__).resolve().parent.parent.parent.parent / "results"
        files = sorted(results_dir.glob("*_human_calibration.json"))
        if files:
            with open(files[-1], "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    @ui.page("/calibration")
    async def calibration_page():
        from qmemory_bench.ui.i18n import t

        dark = ui.dark_mode()
        dark.disable()

        with ui.column().classes("w-full max-w-5xl mx-auto p-6"):

            # Header
            with ui.row().classes("w-full items-center justify-between mb-2"):
                ui.label("🧑‍⚖️ 人工校准").classes("text-3xl font-bold")
                with ui.row().classes("gap-2"):
                    ui.button("← 评测结果", icon="arrow_back",
                              on_click=lambda: ui.navigate.to("/eval")).props("flat dense")
                    ui.button("← 首页", icon="home",
                              on_click=lambda: ui.navigate.to("/")).props("flat dense")

            ui.label("对比 LLM Judge 评分与人工评分，计算校准系数。"
                     "先运行 eval_comprehensive.py --human-calibration 生成校准文件，"
                     "然后在此页面逐题打分。").classes("text-gray-500 mb-4")

            cal_data = _load_calibration_file()
            comp_data = _load_comprehensive_results()

            if not cal_data and not comp_data:
                ui.label("未找到评测结果或校准文件。请先运行评测。").classes("text-red-500 text-xl")
                return

            # If we have comp_data but no cal_data, generate calibration items from comp_data
            if not cal_data and comp_data:
                questions = comp_data.get("questions", [])
                if not questions:
                    ui.label("评测结果中无题目数据。").classes("text-red-500")
                    return
                # Sample up to 30 questions for calibration
                import random
                random.seed(42)
                sample = random.sample(questions, min(30, len(questions)))
                cal_data = {
                    "type": "human_calibration_ui",
                    "items": [
                        {
                            "id": q.get("id", f"cal_{i}"),
                            "category": q.get("category", "general"),
                            "question": q.get("question") or q.get("query", ""),
                            "expected_answer": q.get("expected_answer") or q.get("expected", ""),
                            "system_context_hier": q.get("system_context_hier") or q.get("context_hier", ""),
                            "llm_judge_score": q.get("llm_judge_score", q.get("score_hier", 0)),
                            "llm_judge_reason": q.get("llm_judge_reason", q.get("reason_hier", "")),
                            "llm_judge_precision": q.get("llm_judge_precision", q.get("precision_hier", 0.7)),
                            "human_score": None,
                            "human_precision": None,
                            "human_notes": "",
                        }
                        for i, q in enumerate(sample)
                    ],
                }

            items = cal_data.get("items", [])
            if not items:
                ui.label("校准文件无题目。").classes("text-red-500")
                return

            # Stats
            scored = [it for it in items if it.get("human_score") is not None]
            ui.label(f"共 {len(items)} 题, 已人工评分 {len(scored)} 题").classes("mb-4 text-lg")

            # Score storage
            human_scores: dict[str, dict] = {}
            for it in items:
                human_scores[it["id"]] = {
                    "score": it.get("human_score"),
                    "precision": it.get("human_precision"),
                    "notes": it.get("human_notes", ""),
                }

            # Question cards
            for idx, item in enumerate(items):
                with ui.card().classes("w-full p-4 mb-4"):
                    with ui.row().classes("w-full items-center gap-4 mb-2"):
                        ui.label(f"#{idx+1}").classes("text-2xl font-bold text-gray-400")
                        ui.badge(item["category"]).classes("text-sm")
                        with ui.row().classes("ml-auto gap-2"):
                            ui.label(f"LLM: {item['llm_judge_score']}/10").classes(
                                "text-sm font-mono bg-blue-100 px-2 py-1 rounded")
                            prec = item.get('llm_judge_precision', 0)
                            ui.label(f"P: {prec:.0%}").classes(
                                "text-sm font-mono bg-green-100 px-2 py-1 rounded")

                    ui.label(f"❓ {item['question']}").classes("text-lg font-semibold")
                    ui.label(f"✅ 期望: {item['expected_answer']}").classes("text-sm text-green-700 mt-1")

                    with ui.expansion("📄 系统召回内容").props("dense").classes("mt-2"):
                        ctx = item.get("system_context_hier", "")
                        ui.label(ctx[:1500] if ctx else "(无召回)").classes(
                            "text-xs text-gray-600 whitespace-pre-wrap")

                    if item.get("llm_judge_reason"):
                        ui.label(f"💡 LLM理由: {item['llm_judge_reason']}").classes(
                            "text-xs text-gray-500 mt-1")

                    # Human scoring inputs
                    with ui.row().classes("w-full items-center gap-4 mt-3"):
                        qid = item["id"]

                        score_input = ui.number(
                            "人工评分 (0-10)",
                            min=0, max=10, step=1,
                            value=human_scores[qid]["score"],
                        ).classes("w-32")

                        prec_input = ui.number(
                            "精确率 (0-1)",
                            min=0.0, max=1.0, step=0.1,
                            value=human_scores[qid]["precision"],
                        ).classes("w-32")

                        notes_input = ui.input(
                            "备注",
                            value=human_scores[qid]["notes"],
                        ).classes("flex-1")

                        def make_save(qid=qid, si=score_input, pi=prec_input, ni=notes_input):
                            def save():
                                human_scores[qid]["score"] = si.value
                                human_scores[qid]["precision"] = pi.value
                                human_scores[qid]["notes"] = ni.value
                            return save

                        score_input.on("blur", make_save())
                        prec_input.on("blur", make_save())
                        notes_input.on("blur", make_save())

            # Save & Calculate buttons
            with ui.row().classes("w-full justify-end gap-4 mt-6"):
                status_label = ui.label("").classes("text-sm")

                async def save_calibration():
                    for item in items:
                        hs = human_scores.get(item["id"], {})
                        item["human_score"] = hs.get("score")
                        item["human_precision"] = hs.get("precision")
                        item["human_notes"] = hs.get("notes", "")

                    output = Path(__file__).resolve().parent.parent.parent.parent / "results" / \
                             f"human_calibration_{time.strftime('%Y%m%d_%H%M%S')}.json"
                    output.parent.mkdir(parents=True, exist_ok=True)
                    with open(output, "w", encoding="utf-8") as f:
                        json.dump(cal_data, f, ensure_ascii=False, indent=2)
                    status_label.set_text(f"✅ 已保存: {output.name}")

                async def calc_stats():
                    for item in items:
                        hs = human_scores.get(item["id"], {})
                        item["human_score"] = hs.get("score")
                        item["human_precision"] = hs.get("precision")

                    pairs = [(it["llm_judge_score"], it["human_score"])
                             for it in items if it["human_score"] is not None]
                    if len(pairs) < 3:
                        status_label.set_text("❌ 至少需要3题人工评分才能计算")
                        return

                    # MAE
                    mae_val = sum(abs(p[0]-p[1]) for p in pairs) / len(pairs)
                    avg_llm = sum(p[0] for p in pairs) / len(pairs)
                    avg_human = sum(p[1] for p in pairs) / len(pairs)

                    # Pearson
                    n = len(pairs)
                    sx = sum(p[0] for p in pairs)
                    sy = sum(p[1] for p in pairs)
                    sxy = sum(p[0]*p[1] for p in pairs)
                    sxx = sum(p[0]**2 for p in pairs)
                    syy = sum(p[1]**2 for p in pairs)
                    denom = ((n*sxx - sx*sx) * (n*syy - sy*sy)) ** 0.5
                    r = (n*sxy - sx*sy) / denom if denom else 0

                    status_label.set_text(
                        f"📊 {len(pairs)}题 | LLM均分={avg_llm:.1f} 人工均分={avg_human:.1f} "
                        f"| Pearson r={r:.3f} | MAE={mae_val:.2f}"
                    )

                ui.button("💾 保存评分", on_click=save_calibration).props("color=primary")
                ui.button("📊 计算校准", on_click=calc_stats).props("color=secondary")

    # ── Launch ──
    ui.run(
        port=port,
        title="QMemory Benchmark",
        favicon="📊",
        reload=False,
    )

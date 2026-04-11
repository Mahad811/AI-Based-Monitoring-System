content = open('scripts/demo/demo_server.py', 'r', encoding='utf-8').read()

old = """            # ── Post-alert review hold ──────────────────────────────────────────
            # If an alert fired in this segment, wait for Gemini to complete
            # then give the panel a full REVIEW_HOLD_SECS to read the result
            # before the transition to the next patient begins.
            REVIEW_HOLD_SECS = 10
            if consolidator.fired and seg_idx < len(SEGMENTS) - 1:
                # Wait for any still-running Gemini task (up to 20s)
                # Tasks created via create_task are tracked by asyncio; we poll
                # pending tasks rather than holding a direct reference, so we
                # simply add a generous sleep that covers the API latency.
                # The Gemini job broadcasts its own result when done, so the UI
                # will update automatically — we just need to give it time.
                print(f\"  [Demo] Alert fired — waiting up to 20s for Gemini, then {REVIEW_HOLD_SECS}s review hold\")
                await asyncio.sleep(20)   # covers Gemini T2 + T3 latency

                # Broadcast the review countdown so frontend shows the overlay
                await self.broadcast({
                    \"type\":     \"alert_review\",
                    \"duration\": REVIEW_HOLD_SECS,
                })
                # Tick the countdown so frontend can animate it live
                for remaining in range(REVIEW_HOLD_SECS, 0, -1):
                    await self.broadcast({
                        \"type\":      \"review_tick\",
                        \"remaining\": remaining,
                    })
                    await asyncio.sleep(1)

            # Segment transition pause
            if seg_idx < len(SEGMENTS) - 1:
                await self.broadcast({\"type\": \"transition\", \"message\": \"Switching cameras...\"})
                await asyncio.sleep(GAP_SECONDS)"""

new = """            # ── Post-alert review hold (user-controlled) ────────────────────────
            # If an alert fired, keep the last frame alive and wait until
            # the user clicks "Next Patient" in the navbar.  No fixed timers —
            # the panel drives the pace, and frames keep flowing so the UI
            # never goes blank.
            if consolidator.fired and seg_idx < len(SEGMENTS) - 1:
                print(\"  [Demo] Alert reviewed — holding until user clicks 'Next Patient'\")
                # Notify frontend to show the 'Next Patient' button prominently
                await self.broadcast({\"type\": \"alert_review\", \"duration\": 0})
                self.skip_requested = False   # reset so we wait for a fresh click

                # Keep broadcasting the frozen last frame every 0.5s.
                # Gemini results will appear on their own via broadcast inside
                # execute_gemini_job — we don't need to wait for them here.
                while not self.skip_requested:
                    if last_frame_b64:
                        await self.broadcast({
                            \"type\":         \"frame_update\",
                            \"frame_b64\":    last_frame_b64,
                            \"fall_risk\":    0,
                            \"seizure_risk\": 0,
                            \"fps\":          0,
                        })
                    await asyncio.sleep(0.5)
                self.skip_requested = False   # consume the signal

            # Segment transition pause
            if seg_idx < len(SEGMENTS) - 1:
                await self.broadcast({\"type\": \"transition\", \"message\": \"Switching cameras...\"})
                await asyncio.sleep(GAP_SECONDS)"""

if old in content:
    content = content.replace(old, new, 1)
    open('scripts/demo/demo_server.py', 'w', encoding='utf-8').write(content)
    print('FIXED OK')
else:
    print('Pattern not found — check for CRLF vs LF')
    idx = content.find('Post-alert review hold')
    print(f'Found hold block at char {idx}')
    print(repr(content[idx:idx+200]))

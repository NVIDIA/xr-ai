-- LOVR configuration for the render-mcp scene.
--
-- Kept intentionally minimal — matches the known-working `green-dot` sample
-- (lovr-apps/green-dot/conf.lua). Past attempts to disable extra modules /
-- toggle graphics flags caused `lovr.headset.connect()` to silently fail;
-- treat any deviation from this as a deliberate, tested change.

-- LOVR runs as a child of render-mcp and our stdout is a pipe, not a tty.
-- Without explicit line buffering, every print() is held until the process
-- exits. Force line buffering up front so diagnostics show up live.
io.stdout:setvbuf("line")
io.stderr:setvbuf("line")
print("[render-mcp-scene] conf.lua loaded")

function lovr.conf(t)
    t.headset.drivers = { "openxr" }
    -- Surface lovr.headset.connect() failures (boot.lua:151-155 swallows them
    -- otherwise — they manifest as silent simulator fallback + black frames).
    t.headset.debug = true
    t.window = nil
end

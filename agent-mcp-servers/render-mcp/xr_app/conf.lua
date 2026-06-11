-- LOVR configuration for the render-mcp scene.
--
-- Changes to this file must be tested end-to-end against a headset: module and
-- graphics flags can make `lovr.headset.connect()` fail silently.

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

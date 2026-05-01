-- render-mcp scene.
--
-- Renders a sphere at a fixed world-space position, whose appearance is driven
-- by scene commands arriving from render-mcp (Python) over a ZMQ PULL socket.
--
-- Wire format: msgpack-encoded Lua tables. Recognised ops:
--   { op = "sphere.radius",   value = <number, metres>   }   -- voice-loudness driven
--   { op = "sphere.color",    value = { r, g, b }        }   -- 0..1 floats; speech-driven
--   { op = "sphere.position", value = { x, y, z }        }   -- world-space metres; LLM-driven
--   { op = "sphere.reset"                                 }   -- return colour + position to defaults
--
-- Bridge endpoint comes from the RENDER_SCENE_SOCKET env var (set by render-mcp
-- before it spawns LOVR); a sensible default is used if unset so the file still
-- runs standalone for offline tweaking.

print("[render-mcp-scene] main.lua: top of file")

local zmq = require("lib.zmq")
local mp  = require("lib.msgpack")
print("[render-mcp-scene] main.lua: lib.zmq + lib.msgpack loaded")

-- ── Scene state ───────────────────────────────────────────────────────────────

-- Target radius is what render-mcp tells us we should be at; current radius
-- tracks it with a soft-follow lerp so the visual stays smooth even when
-- commands arrive in bursts.
local target_radius  = 0.05
local current_radius = target_radius

-- Defaults — the canonical look of the sphere at session start. ``sphere.reset``
-- restores these. Keep colour and position in module-local tables so the reset
-- op doesn't need to know magic numbers.
local DEFAULT_COLOR = { 0.2, 0.9, 1.0 }   -- pleasant cyan
local DEFAULT_POS   = { 0.0, 1.6, -1.5 }

-- Same dual-state pattern for colour: the agent emits discrete colour
-- commands (e.g. on STT match), and we ease toward the target so the change
-- looks like a wash rather than a flash.
local target_color  = { DEFAULT_COLOR[1], DEFAULT_COLOR[2], DEFAULT_COLOR[3] }
local current_color = { target_color[1], target_color[2], target_color[3] }

-- Same dual-state pattern for position. Default anchors the sphere at roughly
-- head-height, 1.5 m in front of the origin so a default-posed headset sees it
-- on first connect.
local target_pos  = { DEFAULT_POS[1], DEFAULT_POS[2], DEFAULT_POS[3] }
local current_pos = { target_pos[1], target_pos[2], target_pos[3] }

local radius_lerp = 8.0    -- how fast current_radius follows target_radius
local color_lerp  = 6.0    -- how fast current_color  follows target_color
local pos_lerp    = 6.0    -- how fast current_pos    follows target_pos

-- ── IPC ───────────────────────────────────────────────────────────────────────

local socket_addr = os.getenv("RENDER_SCENE_SOCKET") or "ipc:///tmp/xr_render_scene"
local scene_sock  = nil
local recv_err    = nil

-- Try to connect to the scene bridge.  We keep running even if this fails so
-- the sphere still renders at its default size (makes offline debugging sane).
local ok, err = pcall(function()
    scene_sock = zmq.new_pull_socket(socket_addr)
end)
if not ok then
    recv_err = tostring(err)
    print(string.format("[render-mcp-scene] warning: %s", recv_err))
end

-- ── LOVR callbacks ────────────────────────────────────────────────────────────

function lovr.load()
    -- alpha=0 so the OpenXR runtime composites our framebuffer over the real
    -- environment (passthrough on a real headset, transparent over the page
    -- under IWER) instead of filling the void with black. Requires both:
    --   1) WebXR client requests immersive-ar (so CloudXR exposes ALPHA_BLEND)
    --   2) we explicitly opt into passthrough below — LOVR otherwise picks
    --      blendModes[0], which on CloudXR is OPAQUE.
    lovr.graphics.setBackgroundColor(0.0, 0.0, 0.0, 0.0)
    lovr.headset.setClipDistance(0.1, 256.0)
    print(string.format("[render-mcp-scene] listening on %s", socket_addr))
    -- Confirm OpenXR session actually came up. A printed `isActive=false` here
    -- means LOVR fell back to the desktop simulator (boot.lua:150-155), which
    -- means CloudXR never sees any frames — that's the failure mode we hit.
    local ok_a, active = pcall(lovr.headset.isActive)
    local ok_d, name   = pcall(lovr.headset.getDriver)
    local ok_s, w, h   = pcall(lovr.headset.getDisplayDimensions)
    print(string.format(
        "[render-mcp-scene] headset: active=%s driver=%s display=%s",
        ok_a and tostring(active) or "<err>",
        ok_d and tostring(name)   or "<err>",
        ok_s and string.format("%sx%s", tostring(w), tostring(h)) or "<err>"
    ))

    -- Enumerate + opt into passthrough. Logging both makes it obvious in the
    -- terminal whether the runtime advertised ALPHA_BLEND at all (problem on
    -- the WebXR/CloudXR side) vs. advertised it but rejected our request
    -- (problem on the LOVR/Vulkan side).
    local ok_m, modes = pcall(lovr.headset.getPassthroughModes)
    if ok_m and type(modes) == "table" then
        local parts = {}
        for k, v in pairs(modes) do parts[#parts + 1] = string.format("%s=%s", k, tostring(v)) end
        print("[render-mcp-scene] passthrough modes: " .. table.concat(parts, " "))
    else
        print("[render-mcp-scene] passthrough modes: <err>")
    end
    local ok_p, applied = pcall(lovr.headset.setPassthrough, "blend")
    print(string.format(
        "[render-mcp-scene] setPassthrough('blend') ok=%s result=%s active=%s",
        tostring(ok_p), tostring(applied), tostring(lovr.headset.getPassthrough())
    ))
end

local function drain_commands()
    if not scene_sock then return end
    while true do
        local raw = scene_sock:recv_nonblocking()
        if not raw then break end
        local okd, decoded = pcall(mp.decode, raw)
        if not okd or type(decoded) ~= "table" then goto continue end

        if decoded.op == "sphere.radius" then
            local v = tonumber(decoded.value)
            if v then target_radius = v end

        elseif decoded.op == "sphere.color" then
            local v = decoded.value
            -- Accept arrays {r,g,b} (msgpack-pythonic) and tables {r=…,g=…,b=…}.
            local r = tonumber(v and (v[1] or v.r))
            local g = tonumber(v and (v[2] or v.g))
            local b = tonumber(v and (v[3] or v.b))
            if r and g and b then
                target_color[1] = r
                target_color[2] = g
                target_color[3] = b
            end

        elseif decoded.op == "sphere.position" then
            local v = decoded.value
            -- Same dual-shape acceptance as sphere.color.
            local x = tonumber(v and (v[1] or v.x))
            local y = tonumber(v and (v[2] or v.y))
            local z = tonumber(v and (v[3] or v.z))
            if x and y and z then
                target_pos[1] = x
                target_pos[2] = y
                target_pos[3] = z
            end

        elseif decoded.op == "sphere.reset" then
            -- Restore colour + position to their session-start values. Radius
            -- isn't reset because it's recomputed every audio chunk anyway —
            -- it'll converge on its own once the user goes silent.
            target_color[1], target_color[2], target_color[3] =
                DEFAULT_COLOR[1], DEFAULT_COLOR[2], DEFAULT_COLOR[3]
            target_pos[1], target_pos[2], target_pos[3] =
                DEFAULT_POS[1], DEFAULT_POS[2], DEFAULT_POS[3]
        end

        ::continue::
    end
end

function lovr.update(dt)
    drain_commands()

    -- Smooth the radius a touch so we don't get stutter when commands drop.
    current_radius = current_radius + (target_radius - current_radius) * math.min(1.0, radius_lerp * dt)

    local kc = math.min(1.0, color_lerp * dt)
    current_color[1] = current_color[1] + (target_color[1] - current_color[1]) * kc
    current_color[2] = current_color[2] + (target_color[2] - current_color[2]) * kc
    current_color[3] = current_color[3] + (target_color[3] - current_color[3]) * kc

    local kp = math.min(1.0, pos_lerp * dt)
    current_pos[1] = current_pos[1] + (target_pos[1] - current_pos[1]) * kp
    current_pos[2] = current_pos[2] + (target_pos[2] - current_pos[2]) * kp
    current_pos[3] = current_pos[3] + (target_pos[3] - current_pos[3]) * kp
end

function lovr.draw(pass)
    pass:setColor(current_color[1], current_color[2], current_color[3], 0.95)
    pass:sphere(current_pos[1], current_pos[2], current_pos[3], current_radius)
    pass:setColor(1, 1, 1, 1)

    if recv_err then
        pass:text("render-mcp: " .. recv_err, 0, 2.0, -1.2, 0.06)
    end
end

function lovr.quit()
    if scene_sock then scene_sock:close() end
    return false
end

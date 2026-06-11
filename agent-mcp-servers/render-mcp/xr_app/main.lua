-- SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
-- SPDX-License-Identifier: Apache-2.0

-- render-mcp scene — generic primitive renderer.
--
-- Wire ops (msgpack-encoded tables):
--   { op="scene.add",    value={ id, type, position={x,y,z}, color={r,g,b}, scale } }
--   { op="scene.update", value={ id, [position=…], [color=…], [scale=…] } }
--   { op="scene.remove", value={ id } }

-- ── lovr.log override ────────────────────────────────────────────────────────
-- Emit a tab-separated marker so render-mcp's Python side (the
-- xr_ai_launcher._processes._forward forwarder) can route by authoritative
-- LOVR severity. Levels follow https://lovr.org/docs/lovr.log: debug, info,
-- warn, error. Anything calling lovr.log(...) below (or any LOVR engine
-- message routed through this callback) flows through here.
function lovr.log(message, level, tag)
    if message:sub(-1) == "\n" then message = message:sub(1, -2) end
    io.stdout:write(string.format("LOVR_LOG\t%s\t%s\t%s\n", level, tag or "", message))
    io.stdout:flush()
end

lovr.log("main.lua: top of file", "info", "render-mcp-scene")

local zmq = require("lib.zmq")
local mp  = require("lib.msgpack")
lovr.log("lib.zmq + lib.msgpack loaded", "info", "render-mcp-scene")

-- ── Scene state ───────────────────────────────────────────────────────────────

local primitives      = {}

local pos_lerp   = 6.0
local color_lerp = 6.0
local scale_lerp = 8.0

-- ── IPC ───────────────────────────────────────────────────────────────────────

local socket_addr = os.getenv("RENDER_SCENE_SOCKET") or "ipc:///tmp/xr_render_scene"
local scene_sock  = nil
local recv_err    = nil

local ok, err = pcall(function()
    scene_sock = zmq.new_pull_socket(socket_addr)
end)
if not ok then
    recv_err = tostring(err)
    lovr.log("ZMQ error: " .. recv_err, "info", "render-mcp-scene")
end

-- ── Helpers ───────────────────────────────────────────────────────────────────

local function lerp(a, b, k) return a + (b - a) * k end

local function read_vec3(v, dx, dy, dz)
    return tonumber(v and (v[1] or v.x)) or dx,
           tonumber(v and (v[2] or v.y)) or dy,
           tonumber(v and (v[3] or v.z)) or dz
end

local function make_primitive(ptype, px, py, pz, cr, cg, cb, scale)
    return {
        type          = ptype or "sphere",
        target_pos    = { px, py, pz },
        current_pos   = { px, py, pz },
        target_color  = { cr, cg, cb },
        current_color = { cr, cg, cb },
        target_scale  = scale,
        current_scale = scale,
    }
end

local function count_primitives()
    local n = 0
    for _ in pairs(primitives) do n = n + 1 end
    return n
end

-- ── Op handlers ───────────────────────────────────────────────────────────────

local function handle_scene_add(v)
    local id = v.id
    if not id then
        lovr.log("scene.add: missing id — dropping", "info", "render-mcp-scene")
        return
    end
    local ptype      = v.type or "sphere"
    local px, py, pz = read_vec3(v.position, 0, 1.6, -1.5)
    local cr, cg, cb = read_vec3(v.color, 0.2, 0.9, 1.0)
    local size       = tonumber(v.size) or 0.1
    primitives[id]   = make_primitive(ptype, px, py, pz, cr, cg, cb, size)
    lovr.log(string.format(
        "scene.add  id=%s type=%s pos=(%.2f,%.2f,%.2f) color=(%.2f,%.2f,%.2f) size=%.3fm  total=%d",
        id, ptype, px, py, pz, cr, cg, cb, size, count_primitives()),
        "info", "render-mcp-scene")
end

local function handle_scene_update(v)
    local id  = v.id
    local obj = id and primitives[id]
    if not obj then
        lovr.log(string.format("scene.update: unknown id=%s", tostring(id)),
                 "info", "render-mcp-scene")
        return
    end
    local changed = {}
    if v.position then
        local px, py, pz = read_vec3(v.position, obj.target_pos[1], obj.target_pos[2], obj.target_pos[3])
        obj.target_pos[1], obj.target_pos[2], obj.target_pos[3] = px, py, pz
        changed[#changed+1] = string.format("pos=(%.2f,%.2f,%.2f)", px, py, pz)
    end
    if v.color then
        local cr, cg, cb = read_vec3(v.color, obj.target_color[1], obj.target_color[2], obj.target_color[3])
        obj.target_color[1], obj.target_color[2], obj.target_color[3] = cr, cg, cb
        changed[#changed+1] = string.format("color=(%.2f,%.2f,%.2f)", cr, cg, cb)
    end
    if v.size then
        local s = tonumber(v.size)
        if s then
            obj.target_scale = s
            changed[#changed+1] = string.format("size=%.3fm", s)
        end
    end
    lovr.log(string.format("scene.update id=%s  %s",
                           id, #changed > 0 and table.concat(changed, "  ") or "(nothing changed)"),
             "info", "render-mcp-scene")
end

local function handle_scene_remove(v)
    local id = v.id
    if id and primitives[id] then
        primitives[id] = nil
        lovr.log(string.format("scene.remove id=%s  remaining=%d", id, count_primitives()),
                 "info", "render-mcp-scene")
    elseif id then
        lovr.log(string.format("scene.remove: unknown id=%s", id),
                 "info", "render-mcp-scene")
    end
end

-- ── LOVR callbacks ────────────────────────────────────────────────────────────

function lovr.load()
    lovr.graphics.setBackgroundColor(0, 0, 0, 0)
    lovr.headset.setClipDistance(256.0, 0.15)
    lovr.log("lovr.load  socket=" .. socket_addr, "info", "render-mcp-scene")

    local ok_a, active = pcall(lovr.headset.isActive)
    local ok_d, name   = pcall(lovr.headset.getDriver)
    lovr.log(string.format("headset: active=%s driver=%s",
        ok_a and tostring(active) or "<err>",
        ok_d and tostring(name)   or "<err>"),
        "info", "render-mcp-scene")

    local ok_p, applied = pcall(lovr.headset.setPassthrough, "blend")
    lovr.log(string.format("setPassthrough('blend') ok=%s applied=%s",
        tostring(ok_p), tostring(applied)),
        "info", "render-mcp-scene")
end

local function drain_commands()
    if not scene_sock then return end
    while true do
        local raw = scene_sock:recv_nonblocking()
        if not raw then break end

        local okd, decoded = pcall(mp.decode, raw)
        if not okd then
            lovr.log("msgpack decode error: " .. tostring(decoded),
                     "info", "render-mcp-scene")
            goto continue
        end
        if type(decoded) ~= "table" then
            lovr.log("unexpected message type: " .. type(decoded),
                     "info", "render-mcp-scene")
            goto continue
        end

        local op = decoded.op
        local v  = decoded.value or {}

        -- Wrap each handler in pcall so one bad message can't crash lovr.update.
        local ok_h, herr
        if     op == "scene.add"    then ok_h, herr = pcall(handle_scene_add,    v)
        elseif op == "scene.update" then ok_h, herr = pcall(handle_scene_update, v)
        elseif op == "scene.remove" then ok_h, herr = pcall(handle_scene_remove, v)
        elseif op ~= nil then
            lovr.log("unknown op=" .. tostring(op), "info", "render-mcp-scene")
        else
            lovr.log("message missing 'op' field", "info", "render-mcp-scene")
        end

        if ok_h == false then
            lovr.log(string.format("handler error (op=%s): %s",
                                   tostring(op), tostring(herr)),
                     "info", "render-mcp-scene")
        end

        ::continue::
    end
end

local heartbeat_t = 0.0

function lovr.update(dt)
    drain_commands()

    local ks = math.min(1.0, scale_lerp * dt)
    local kc = math.min(1.0, color_lerp * dt)
    local kp = math.min(1.0, pos_lerp   * dt)

    for _, obj in pairs(primitives) do
        obj.current_scale    = lerp(obj.current_scale,    obj.target_scale,    ks)
        obj.current_color[1] = lerp(obj.current_color[1], obj.target_color[1], kc)
        obj.current_color[2] = lerp(obj.current_color[2], obj.target_color[2], kc)
        obj.current_color[3] = lerp(obj.current_color[3], obj.target_color[3], kc)
        obj.current_pos[1]   = lerp(obj.current_pos[1],   obj.target_pos[1],   kp)
        obj.current_pos[2]   = lerp(obj.current_pos[2],   obj.target_pos[2],   kp)
        obj.current_pos[3]   = lerp(obj.current_pos[3],   obj.target_pos[3],   kp)
    end

    heartbeat_t = heartbeat_t + dt
    if heartbeat_t >= 5.0 then
        heartbeat_t = 0.0
        local n = count_primitives()
        if n == 0 then
            lovr.log("heartbeat: scene empty", "info", "render-mcp-scene")
        else
            for id, obj in pairs(primitives) do
                lovr.log(string.format(
                    "heartbeat: id=%s type=%s pos=(%.2f,%.2f,%.2f) size=%.3fm",
                    id, obj.type,
                    obj.current_pos[1], obj.current_pos[2], obj.current_pos[3],
                    obj.current_scale),
                    "info", "render-mcp-scene")
            end
        end
    end
end

function lovr.draw(pass)
    for _, obj in pairs(primitives) do
        pass:setColor(obj.current_color[1], obj.current_color[2], obj.current_color[3], 0.95)
        local p, s = obj.current_pos, obj.current_scale
        if obj.type == "box" then
            pass:box(p[1], p[2], p[3], s, s, s)
        else
            pass:sphere(p[1], p[2], p[3], s)
        end
    end
    pass:setColor(1, 1, 1, 1)
    if recv_err then
        pass:text("ZMQ error: " .. recv_err, 0, 2.0, -1.2, 0.06)
    end
end

function lovr.quit()
    if scene_sock then scene_sock:close() end
    return false
end

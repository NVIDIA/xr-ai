-- Minimal msgpack decoder for LuaJIT (render-mcp scene bridge).
--
-- Covers the subset emitted by Python's msgpack.packb(..., use_bin_type=True):
--   fixmap, fixarray, fixstr, fixint, nil, bool,
--   float32, float64, uint8/16/32, int8/16/32,
--   str8/16/32, array16/32, map16/32.
--
-- Float bytes are reinterpreted via FFI type-punning (no manual IEEE 754 math).
--
-- Public API:
--   msgpack.decode(s) → value

local ffi  = require("ffi")
local byte = string.byte
local sub  = string.sub

-- ── float helpers via FFI type-punning ───────────────────────────────────────

local _f32_buf = ffi.new("uint8_t[4]")
local _f64_buf = ffi.new("uint8_t[8]")
local _f32_ptr = ffi.cast("float  *", _f32_buf)
local _f64_ptr = ffi.cast("double *", _f64_buf)

local function f32(s, i)
    -- msgpack is big-endian; x86 is little-endian
    _f32_buf[3], _f32_buf[2], _f32_buf[1], _f32_buf[0] = byte(s, i, i + 3)
    return tonumber(_f32_ptr[0])
end

local function f64(s, i)
    _f64_buf[7], _f64_buf[6], _f64_buf[5], _f64_buf[4],
    _f64_buf[3], _f64_buf[2], _f64_buf[1], _f64_buf[0] = byte(s, i, i + 7)
    return tonumber(_f64_ptr[0])
end

-- ── integer helpers ───────────────────────────────────────────────────────────

local function u8(s, i)  return byte(s, i) end
local function u16(s, i) local a, b = byte(s, i, i + 1); return a * 256 + b end
local function u32(s, i)
    local a, b, c, d = byte(s, i, i + 3)
    return ((a * 256 + b) * 256 + c) * 256 + d
end

-- ── recursive decoder ─────────────────────────────────────────────────────────

local function decode_one(s, pos)
    local b = u8(s, pos)

    -- positive fixint  0x00–0x7f
    if b <= 0x7f then return b, pos + 1 end

    -- fixmap  0x80–0x8f
    if b <= 0x8f then
        local n, t = b - 0x80, {}
        pos = pos + 1
        for _ = 1, n do
            local k; k, pos = decode_one(s, pos)
            local v; v, pos = decode_one(s, pos)
            t[k] = v
        end
        return t, pos
    end

    -- fixarray  0x90–0x9f
    if b <= 0x9f then
        local n, t = b - 0x90, {}
        pos = pos + 1
        for i = 1, n do t[i], pos = decode_one(s, pos) end
        return t, pos
    end

    -- fixstr  0xa0–0xbf
    if b <= 0xbf then
        local n = b - 0xa0
        return sub(s, pos + 1, pos + n), pos + 1 + n
    end

    -- nil / false / true
    if b == 0xc0 then return nil,   pos + 1 end
    if b == 0xc2 then return false, pos + 1 end
    if b == 0xc3 then return true,  pos + 1 end

    -- bin8 (decode as Lua string)
    if b == 0xc4 then
        local n = u8(s, pos + 1)
        return sub(s, pos + 2, pos + 1 + n), pos + 2 + n
    end

    -- float32 / float64
    if b == 0xca then return f32(s, pos + 1), pos + 5 end
    if b == 0xcb then return f64(s, pos + 1), pos + 9 end

    -- uint8 / uint16 / uint32
    if b == 0xcc then return u8(s,  pos + 1), pos + 2 end
    if b == 0xcd then return u16(s, pos + 1), pos + 3 end
    if b == 0xce then return u32(s, pos + 1), pos + 5 end

    -- int8 / int16 / int32
    if b == 0xd0 then
        local v = u8(s, pos + 1)
        return (v >= 128 and v - 256 or v), pos + 2
    end
    if b == 0xd1 then
        local v = u16(s, pos + 1)
        return (v >= 32768 and v - 65536 or v), pos + 3
    end
    if b == 0xd2 then
        local v = u32(s, pos + 1)
        return (v >= 2147483648.0 and v - 4294967296.0 or v), pos + 5
    end

    -- str8 / str16 / str32
    if b == 0xd9 then
        local n = u8(s, pos + 1)
        return sub(s, pos + 2, pos + 1 + n), pos + 2 + n
    end
    if b == 0xda then
        local n = u16(s, pos + 1)
        return sub(s, pos + 3, pos + 2 + n), pos + 3 + n
    end
    if b == 0xdb then
        local n = u32(s, pos + 1)
        return sub(s, pos + 5, pos + 4 + n), pos + 5 + n
    end

    -- array16 / array32
    if b == 0xdc then
        local n, t = u16(s, pos + 1), {}
        pos = pos + 3
        for i = 1, n do t[i], pos = decode_one(s, pos) end
        return t, pos
    end
    if b == 0xdd then
        local n, t = u32(s, pos + 1), {}
        pos = pos + 5
        for i = 1, n do t[i], pos = decode_one(s, pos) end
        return t, pos
    end

    -- map16 / map32
    if b == 0xde then
        local n, t = u16(s, pos + 1), {}
        pos = pos + 3
        for _ = 1, n do
            local k; k, pos = decode_one(s, pos)
            local v; v, pos = decode_one(s, pos)
            t[k] = v
        end
        return t, pos
    end
    if b == 0xdf then
        local n, t = u32(s, pos + 1), {}
        pos = pos + 5
        for _ = 1, n do
            local k; k, pos = decode_one(s, pos)
            local v; v, pos = decode_one(s, pos)
            t[k] = v
        end
        return t, pos
    end

    -- negative fixint  0xe0–0xff
    if b >= 0xe0 then return b - 256, pos + 1 end

    error(string.format("msgpack: unsupported type 0x%02x at pos %d", b, pos))
end

local M = {}

function M.decode(s)
    local val = decode_one(s, 1)
    return val
end

return M

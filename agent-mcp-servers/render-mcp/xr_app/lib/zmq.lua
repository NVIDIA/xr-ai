-- LuaJIT FFI wrapper for ZMQ PULL sockets.
--
-- render-mcp sets RENDER_ZMQ_LIB to the libzmq bundled with pyzmq before
-- spawning LOVR; we fall back to the system library if it is not set.
--
-- Public API:
--   zmq.new_pull_socket(addr) → socket
--   socket:recv_nonblocking()  → string or nil
--   socket:close()

local ffi = require("ffi")

ffi.cdef[[
    void *zmq_ctx_new(void);
    void *zmq_socket(void *context, int type);
    int   zmq_connect(void *socket, const char *endpoint);
    int   zmq_recv(void *socket, void *buf, size_t len, int flags);
    int   zmq_close(void *socket);
    int   zmq_errno(void);
    const char *zmq_strerror(int errnum);
]]

local ZMQ_PULL     = 7
local ZMQ_DONTWAIT = 1
local BUF_SIZE     = 65536

local lib_path = os.getenv("RENDER_ZMQ_LIB") or "zmq"
local zmq_lib  = ffi.load(lib_path)

local _ctx = zmq_lib.zmq_ctx_new()
assert(_ctx ~= nil, "zmq_ctx_new failed")

local M = {}

function M.new_pull_socket(addr)
    local sock = zmq_lib.zmq_socket(_ctx, ZMQ_PULL)
    if sock == nil then
        error("zmq_socket failed: " ..
              ffi.string(zmq_lib.zmq_strerror(zmq_lib.zmq_errno())))
    end
    local rc = zmq_lib.zmq_connect(sock, addr)
    if rc ~= 0 then
        zmq_lib.zmq_close(sock)
        error("zmq_connect to " .. addr .. " failed: " ..
              ffi.string(zmq_lib.zmq_strerror(zmq_lib.zmq_errno())))
    end

    local _buf = ffi.new("uint8_t[?]", BUF_SIZE)
    local obj  = {}

    function obj:recv_nonblocking()
        local n = zmq_lib.zmq_recv(sock, _buf, BUF_SIZE, ZMQ_DONTWAIT)
        if n < 0 then return nil end
        return ffi.string(_buf, n)
    end

    function obj:close()
        zmq_lib.zmq_close(sock)
    end

    return obj
end

return M

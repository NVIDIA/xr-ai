package com.nvidia.xrai.streamkitsample.streamkit

/**
 * Connection lifecycle state reported by [StreamSession].
 *
 * Mirror of Swift `ConnectionState` and the web `ConnectionState` frozen enum.
 */
enum class ConnectionState {
    DISCONNECTED,
    CONNECTING,
    CONNECTED,
    RECONNECTING,
}

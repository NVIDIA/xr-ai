import Foundation

/// Connection lifecycle state reported by ``StreamSession``.
public enum ConnectionState: Sendable, Equatable {
    case disconnected
    case connecting
    case connected
    case reconnecting
}

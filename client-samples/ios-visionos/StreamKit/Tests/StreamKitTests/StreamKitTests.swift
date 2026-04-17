import Testing
@testable import StreamKit

@Suite("ConnectionState")
struct ConnectionStateTests {
    @Test func equatable() {
        #expect(ConnectionState.connected == .connected)
        #expect(ConnectionState.disconnected != .connected)
    }
}

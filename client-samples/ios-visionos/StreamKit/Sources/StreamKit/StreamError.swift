import Foundation

/// Errors thrown by ``StreamSession`` and its backends.
public enum StreamError: Error, LocalizedError, Sendable {

    /// Host string could not be turned into a valid URL.
    case invalidHost(String)

    /// An operation that requires an active connection was called while disconnected.
    case notConnected

    /// Neither a `token` nor a `tokenURL` was provided to the LiveKit backend.
    case missingToken

    /// Token-server request failed or returned an unparseable body.
    case tokenFetchFailed(URL)

    /// `startCamera()` was called while not connected.
    case cameraRequiresConnection

    /// *(visionOS)* `startCamera()` was called but no immersive space is open.
    ///
    /// Open your app's `ImmersiveSpace` scene **before** calling `startCamera()`.
    case immersiveSpaceRequired

    // MARK: - LocalizedError

    public var errorDescription: String? {
        switch self {
        case .invalidHost(let h):         return "'\(h)' is not a valid hostname."
        case .notConnected:               return "Not connected. Call connect() first."
        case .missingToken:               return "Provide a token or tokenURL in LiveKitConfig."
        case .tokenFetchFailed(let url):  return "Failed to fetch token from \(url)."
        case .cameraRequiresConnection:   return "Connect before starting the camera."
        case .immersiveSpaceRequired:
            return "Camera capture on visionOS requires an open ImmersiveSpace. " +
                   "Call openImmersiveSpace() in your app before startCamera()."
        }
    }
}

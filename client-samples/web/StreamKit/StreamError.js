/**
 * @fileoverview Errors thrown by {@link StreamSession} and its backends.
 *
 * Mirror of the Swift `StreamError` enum. Each named case maps to a static
 * factory method that returns a `StreamError` instance with a user-friendly
 * `message` property matching the Swift `errorDescription`.
 *
 * @module StreamKit/StreamError
 */

/**
 * Discriminated error type for StreamKit operations.
 *
 * Do not construct directly — use the static factory methods so that
 * `instanceof StreamError` and the `code` field work correctly.
 *
 * @example
 * import { StreamError } from './StreamKit/StreamError.js';
 *
 * try {
 *   await session.connect(config);
 * } catch (err) {
 *   if (err instanceof StreamError) {
 *     console.error(err.code, err.message);
 *   }
 * }
 */
export class StreamError extends Error {
  /**
   * @param {string} code     - Machine-readable case name (e.g. `'invalidHost'`).
   * @param {string} message  - User-facing description.
   * @param {unknown} [cause] - Optional underlying error.
   */
  constructor(code, message, cause) {
    super(message);
    this.name = 'StreamError';
    /** @type {string} Machine-readable error code matching the Swift case name. */
    this.code = code;
    if (cause !== undefined) this.cause = cause;
  }

  // -------------------------------------------------------------------------
  // Static factories — one per Swift named case
  // -------------------------------------------------------------------------

  /**
   * Host string could not be turned into a valid URL.
   *
   * @param {string} host - The invalid host string.
   * @returns {StreamError}
   */
  static invalidHost(host) {
    return new StreamError('invalidHost', `'${host}' is not a valid hostname.`);
  }

  /**
   * An operation that requires an active connection was called while disconnected.
   *
   * @returns {StreamError}
   */
  static notConnected() {
    return new StreamError('notConnected', 'Not connected. Call connect() first.');
  }

  /**
   * Neither a `token` nor a `tokenURL` was provided to the LiveKit backend.
   *
   * @returns {StreamError}
   */
  static missingToken() {
    return new StreamError('missingToken', 'Provide a token or tokenURL in LiveKitConfig.');
  }

  /**
   * Token-server request failed or returned an unparseable body.
   *
   * @param {string|URL} url - The token endpoint that failed.
   * @returns {StreamError}
   */
  static tokenFetchFailed(url) {
    return new StreamError('tokenFetchFailed', `Failed to fetch token from ${url}.`);
  }

  /**
   * `startCamera()` was called while not connected.
   *
   * @returns {StreamError}
   */
  static cameraRequiresConnection() {
    return new StreamError('cameraRequiresConnection', 'Connect before starting the camera.');
  }
}

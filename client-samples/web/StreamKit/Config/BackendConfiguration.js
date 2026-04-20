/**
 * @fileoverview Backend selection and LiveKit connection parameters.
 *
 * Mirror of the Swift `BackendConfiguration` enum and `LiveKitConfig` struct.
 * `BackendConfiguration` is a discriminated class with a static factory for
 * each built-in backend. `makeBackend()` dynamically imports the concrete
 * implementation to keep the core library tree-shakeable.
 *
 * @module StreamKit/Config/BackendConfiguration
 */

// ─────────────────────────────────────────────────────────────────────────────
// LiveKitConfig
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Connection parameters for the LiveKit backend.
 *
 * Exactly one of {@link LiveKitConfig#token} or {@link LiveKitConfig#tokenURL}
 * must be provided before calling `StreamSession.connect()`.
 *
 * @example
 * // Static token (pre-signed JWT)
 * const cfg = new LiveKitConfig({ host: '192.168.1.100', token: myJwt });
 *
 * // Token endpoint (the SDK appends ?room=…&identity=… automatically)
 * const cfg = new LiveKitConfig({ host: '192.168.1.100', tokenURL: 'https://…/token' });
 */
export class LiveKitConfig {
  /**
   * IP address or hostname of the LiveKit server (e.g. `"192.168.1.100"`).
   * Do not include a scheme or port — those are handled by {@link LiveKitConfig#port}
   * and {@link LiveKitConfig#secure}.
   *
   * @type {string}
   */
  host;

  /**
   * WebSocket port. Defaults to `7880` (LiveKit's default).
   *
   * @type {number}
   */
  port;

  /**
   * Use `wss://` / `https://`. Set to `false` for local / LAN connections.
   *
   * @type {boolean}
   */
  secure;

  /**
   * A pre-signed LiveKit JWT token.
   * The token must encode the room name and participant identity.
   *
   * @type {string|null}
   */
  token;

  /**
   * URL string of a token-generation endpoint.
   *
   * The SDK appends `?room=xr-room&identity=<identity>` query parameters.
   * The endpoint must return either a plain JWT string or `{ "token": "eyJ…" }`.
   *
   * @type {string|null}
   */
  tokenURL;

  /**
   * @param {object}      opts
   * @param {string}      opts.host
   * @param {number}      [opts.port=7880]
   * @param {boolean}     [opts.secure=false]
   * @param {string|null} [opts.token=null]
   * @param {string|null} [opts.tokenURL=null]
   */
  constructor({ host, port = 7880, secure = false, token = null, tokenURL = null }) {
    this.host = host;
    this.port = port;
    this.secure = secure;
    this.token = token;
    this.tokenURL = tokenURL;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// BackendConfiguration
// ─────────────────────────────────────────────────────────────────────────────

/** @typedef {'liveKit'} BackendKind */

/**
 * Selects the networking backend used by `StreamSession`.
 *
 * Pass an instance to `new StreamSession(backendConfig)` to use a built-in
 * backend. To supply a completely custom implementation, pass your own object
 * that satisfies the {@link StreamingBackend} interface directly.
 *
 * ```js
 * // Built-in LiveKit backend
 * const session = new StreamSession(
 *   BackendConfiguration.liveKit(new LiveKitConfig({ host: '192.168.1.100', token: jwt }))
 * );
 *
 * // Custom backend (e.g. your own streaming SDK)
 * const session = new StreamSession(new MyCustomBackend());
 * ```
 */
export class BackendConfiguration {
  /** @type {BackendKind} */
  #kind;

  /** @type {LiveKitConfig} */
  #config;

  /**
   * @param {BackendKind}   kind
   * @param {LiveKitConfig} config
   */
  constructor(kind, config) {
    this.#kind = kind;
    this.#config = config;
  }

  // -------------------------------------------------------------------------
  // Static factories
  // -------------------------------------------------------------------------

  /**
   * The LiveKit WebRTC backend.
   *
   * @param {LiveKitConfig} config
   * @returns {BackendConfiguration}
   */
  static liveKit(config) {
    return new BackendConfiguration('liveKit', config);
  }

  // -------------------------------------------------------------------------
  // Factory
  // -------------------------------------------------------------------------

  /**
   * Instantiates the concrete {@link StreamingBackend} for this configuration.
   *
   * Uses a dynamic import so the `LiveKitBackend` module (and the LiveKit JS
   * SDK it bundles) is only fetched when actually needed — mirroring the
   * Swift `makeBackend()` method on `BackendConfiguration`.
   *
   * @returns {Promise<import('../Backends/StreamingBackend.js').StreamingBackend>}
   */
  async makeBackend() {
    switch (this.#kind) {
      case 'liveKit': {
        const { LiveKitBackend } = await import('../Backends/LiveKit/LiveKitBackend.js');
        return new LiveKitBackend(this.#config);
      }
      default:
        throw new Error(`Unknown backend kind: ${this.#kind}`);
    }
  }
}

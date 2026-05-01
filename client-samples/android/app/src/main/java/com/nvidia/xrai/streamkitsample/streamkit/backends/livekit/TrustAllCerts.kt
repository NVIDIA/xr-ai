// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.xrai.streamkitsample.streamkit.backends.livekit

import okhttp3.OkHttpClient
import java.security.SecureRandom
import java.security.cert.X509Certificate
import javax.net.ssl.HostnameVerifier
import javax.net.ssl.SSLContext
import javax.net.ssl.SSLSocketFactory
import javax.net.ssl.X509TrustManager

/**
 * Trust-everything helpers for talking to the xr-ai hub's self-signed
 * TLS certificate without requiring the user to manually install a profile.
 *
 * **This is appropriate for a development sample only.** Production apps
 * must pin the server certificate or rely on a real CA — accepting any
 * cert defeats the protection TLS is meant to provide.
 *
 * Mirrors the browser's "Advanced → Proceed to … (unsafe)" affordance and
 * the iOS sample's similar dev-only setup.
 */
internal object TrustAllCerts {

    private val trustManager = object : X509TrustManager {
        override fun checkClientTrusted(chain: Array<X509Certificate>?, authType: String?) {
            // dev-only: hub uses a self-signed cert; not for production use // NOSONAR
        }
        override fun checkServerTrusted(chain: Array<X509Certificate>?, authType: String?) {
            // dev-only: hub uses a self-signed cert; not for production use // NOSONAR
        }
        override fun getAcceptedIssuers(): Array<X509Certificate> = emptyArray()
    }

    private val hostnameVerifier = HostnameVerifier { _, _ -> true }

    private val sslContext: SSLContext = SSLContext.getInstance("TLSv1.2").apply {
        init(null, arrayOf<javax.net.ssl.TrustManager>(trustManager), SecureRandom())
    }

    /** SSLSocketFactory for HttpsURLConnection. */
    val socketFactory: SSLSocketFactory = sslContext.socketFactory

    /** Permissive hostname verifier — pairs with [socketFactory] for HttpsURLConnection. */
    val permissiveHostnameVerifier: HostnameVerifier = hostnameVerifier

    /** OkHttpClient configured to trust any cert — pass to LiveKitOverrides. */
    fun okHttpClient(): OkHttpClient = OkHttpClient.Builder()
        .sslSocketFactory(sslContext.socketFactory, trustManager)
        .hostnameVerifier(hostnameVerifier)
        .build()
}

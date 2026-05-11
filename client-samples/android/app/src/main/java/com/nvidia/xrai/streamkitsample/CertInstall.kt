// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

package com.nvidia.xrai.streamkitsample

import android.content.Intent
import android.security.KeyChain
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import java.net.URL
import java.security.SecureRandom
import java.security.cert.CertificateFactory
import java.security.cert.X509Certificate
import javax.net.ssl.HttpsURLConnection
import javax.net.ssl.SSLContext
import javax.net.ssl.X509TrustManager

/**
 * Fetches the hub's self-signed CA cert and returns a [KeyChain] install [Intent], or null on
 * failure. The fetch uses a single-connection trust-all SSLSocketFactory because the cert being
 * downloaded is the one that is not yet trusted.
 */
internal suspend fun fetchCertInstallIntent(
    host: String,
    port: String,
): Intent? = withContext(Dispatchers.IO) {
    val url = "https://$host:$port/cert"
    val conn = (URL(url).openConnection() as HttpsURLConnection).apply {
        connectTimeout = 8_000
        readTimeout    = 8_000
        // Single-connection trust-all: the cert being fetched is exactly the one
        // we don't yet trust, so normal validation cannot succeed here.
        val trustAll = object : X509TrustManager {
            override fun checkClientTrusted(c: Array<out X509Certificate>?, t: String?) = Unit
            override fun checkServerTrusted(c: Array<out X509Certificate>?, t: String?) = Unit // NOSONAR
            override fun getAcceptedIssuers(): Array<X509Certificate> = emptyArray()
        }
        val ctx = SSLContext.getInstance("TLSv1.2").apply {
            init(null, arrayOf(trustAll), SecureRandom())
        }
        sslSocketFactory  = ctx.socketFactory
        hostnameVerifier  = javax.net.ssl.HostnameVerifier { _, _ -> true }
    }
    try {
        val bytes = conn.inputStream.use { it.readBytes() }
        CertificateFactory.getInstance("X.509").generateCertificate(bytes.inputStream())
        KeyChain.createInstallIntent().apply {
            putExtra(KeyChain.EXTRA_CERTIFICATE, bytes)
            putExtra(KeyChain.EXTRA_NAME, "xr-ai-hub")
        }
    } catch (_: Exception) {
        null
    } finally {
        conn.disconnect()
    }
}

// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import java.util.Properties
import java.util.jar.JarFile

// Plugin classpaths resolve before the main script body. Install both rules
// during the buildscript phase, before AGP can load its dependencies.
buildscript {
    // AGP's UTP test tooling pulls io.netty 4.1.93.Final (via grpc-netty), which has known
    // CVEs; lift anything older to the pinned version. Buildscript and project configurations
    // resolve separately, so both need the rule.
    val catalog = project.extensions.getByType<VersionCatalogsExtension>().named("libs")
    val nettyPin = catalog.findVersion("netty").get().requiredVersion
    val guavaPin = catalog.findVersion("guava").get().requiredVersion
    val nettyPinParts = nettyPin.split('.').mapNotNull { it.toIntOrNull() }

    fun belowNettyPin(version: String): Boolean {
        if (!version.startsWith("4.")) return false
        val parts = version.split('.').mapNotNull { it.toIntOrNull() }
        nettyPinParts.forEachIndexed { i, pin ->
            val requested = parts.getOrElse(i) { 0 }
            if (requested != pin) return requested < pin
        }
        return false
    }

    fun Configuration.applySecurityVersions() = resolutionStrategy.eachDependency {
        if (requested.group == "io.netty" && belowNettyPin(requested.version.orEmpty())) {
            useVersion(nettyPin)
        }
        // CVE-2026-102554: avoid eager allocation during Guava deserialization.
        if (requested.group == "com.google.guava" && requested.name == "guava") {
            useVersion(guavaPin)
        }
    }

    allprojects {
        buildscript.configurations.configureEach { applySecurityVersions() }
        configurations.configureEach { applySecurityVersions() }
    }

    // CI runs this so the lift fails loudly if a Gradle or AGP change stops it applying.
    tasks.register("verifyNettyPin") {
        doLast {
            val offenders = allprojects.flatMap { p ->
                (p.buildscript.configurations.toList() +
                    p.configurations.filter { it.name.contains("unified-test-platform") })
                    .filter { it.isCanBeResolved }
                    .flatMap { c ->
                        runCatching { c.resolvedConfiguration.lenientConfiguration.allModuleDependencies }
                            .getOrDefault(emptySet())
                            .filter { it.moduleGroup == "io.netty" && belowNettyPin(it.moduleVersion) }
                            .map { "${c.name}: ${it.moduleGroup}:${it.moduleName}:${it.moduleVersion}" }
                    }
            }
            check(offenders.isEmpty()) { "io.netty below $nettyPin:\n" + offenders.joinToString("\n") }
        }
    }
}

plugins {
    alias(libs.plugins.android.application) apply false
    alias(libs.plugins.kotlin.android) apply false
    alias(libs.plugins.kotlin.compose) apply false
}

// Opt-in only: the locks under dependency-manifest/android are a cutoff snapshot
// for dependency analysis, not a build input (see DEPENDENCIES.md). The Kotlin
// plugin's ABI validation classpath resolves a floating version, so it stays out.
if (gradle.startParameter.isWriteDependencyLocks) {
    val lockDir = rootDir.resolve("../../dependency-manifest/android").normalize()
    buildscript.dependencyLocking {
        lockAllConfigurations()
        lockFile.set(lockDir.resolve("buildscript-gradle.lockfile"))
    }
    project(":app") {
        dependencyLocking {
            lockAllConfigurations()
            lockFile.set(lockDir.resolve("gradle.lockfile"))
        }
        configurations.matching { it.name.startsWith("kotlinAbiValidation") }.configureEach {
            resolutionStrategy.deactivateDependencyLocking()
        }
    }
}

// Read metadata from the JARs AGP actually loads; a resolved dependency report
// can show newer versions even when the plugin classloader still uses old ones.
tasks.register("verifyBuildscriptSecurityVersions") {
    doLast {
        val agpClassLoader = project(":app").plugins.getPlugin("com.android.application").javaClass.classLoader
        val expectedVersions = listOf(
            Triple("com.google.common.collect.ImmutableList", "com.google.guava/guava", libs.versions.guava.get()),
            Triple("io.netty.util.Version", "io.netty/netty-common", libs.versions.netty.get()),
        )
        expectedVersions.forEach { (className, artifact, expectedVersion) ->
            val source = agpClassLoader.loadClass(className).protectionDomain.codeSource.location
            val loadedVersion = JarFile(File(source.toURI())).use { jar ->
                val metadata = jar.getJarEntry("META-INF/maven/$artifact/pom.properties")
                val properties = Properties()
                jar.getInputStream(metadata).use { properties.load(it) }
                properties.getProperty("version")
            }
            check(loadedVersion == expectedVersion) {
                "AGP loaded $artifact:$loadedVersion from $source; expected $expectedVersion"
            }
            logger.lifecycle("AGP loaded $artifact:$loadedVersion from $source")
        }
    }
}

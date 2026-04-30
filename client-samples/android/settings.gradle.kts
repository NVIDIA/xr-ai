pluginManagement {
    repositories {
        google {
            content {
                includeGroupByRegex("com\\.android.*")
                includeGroupByRegex("com\\.google.*")
                includeGroupByRegex("androidx.*")
            }
        }
        mavenCentral()
        gradlePluginPortal()
    }
}
dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
        // LiveKit Android transitively depends on `com.github.davidliu:audioswitch`
        // pinned to a Git commit hash, which is only published on JitPack.
        maven { url = uri("https://jitpack.io") }
    }
}

rootProject.name = "StreamKitSample"
include(":app")

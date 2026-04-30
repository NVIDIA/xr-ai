# LiveKit — keep all classes used by reflection
-keep class io.livekit.** { *; }
-dontwarn io.livekit.**

# WebRTC native symbols
-keep class org.webrtc.** { *; }
-dontwarn org.webrtc.**

# OkHttp / okio used transitively by LiveKit
-dontwarn okhttp3.**
-dontwarn okio.**

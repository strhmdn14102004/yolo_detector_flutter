# TensorFlow Lite GPU Delegate
-keep class org.tensorflow.** { *; }
-dontwarn org.tensorflow.**

# Google MLKit Face Detection
-keep class com.google.mlkit.** { *; }
-dontwarn com.google.mlkit.**

# CameraX
-keep class androidx.camera.** { *; }
-dontwarn androidx.camera.**

# Flutter Plugins
-keep class io.flutter.** { *; }
-dontwarn io.flutter.**

# Prevent removing annotations (for ML Kit)
-keepattributes *Annotation*

param(
    [string]$Serial = "emulator-5554"
)

$ErrorActionPreference = "Stop"
$workspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot "../../../..")).Path
$sdkRoot = if ($env:ANDROID_HOME) { $env:ANDROID_HOME } else { $env:ANDROID_SDK_ROOT }
if (-not $sdkRoot) { throw "Set ANDROID_HOME or ANDROID_SDK_ROOT." }

$ndkRoot = if ($env:ANDROID_NDK_HOME) { $env:ANDROID_NDK_HOME } else {
    Get-ChildItem (Join-Path $sdkRoot "ndk") -Directory | Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty FullName
}
if (-not $ndkRoot) { throw "No Android NDK was found under $sdkRoot." }
$hostToolchain = Join-Path $ndkRoot "toolchains/llvm/prebuilt/windows-x86_64/bin"
$env:CARGO_TARGET_X86_64_LINUX_ANDROID_LINKER = Join-Path $hostToolchain "x86_64-linux-android29-clang.cmd"

$buildTools = Get-ChildItem (Join-Path $sdkRoot "build-tools") -Directory | Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty FullName
$platform = Get-ChildItem (Join-Path $sdkRoot "platforms") -Directory | Where-Object { Test-Path (Join-Path $_.FullName "android.jar") } | Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty FullName
$androidJar = Join-Path $platform "android.jar"
$out = Join-Path $workspaceRoot "target/android-01-triangle"
$staging = Join-Path $out "staging"
$nativeDir = Join-Path $staging "lib/x86_64"
New-Item -ItemType Directory -Force -Path $nativeDir | Out-Null

Push-Location $workspaceRoot
try {
    cargo build -p fluxel-android-triangle --target x86_64-linux-android --no-default-features --features vulkan --release
    if ($LASTEXITCODE -ne 0) { throw "Android triangle build failed." }
} finally {
    Pop-Location
}

$nativeLibrary = Join-Path $workspaceRoot "target/x86_64-linux-android/release/libfluxel_android_triangle.so"
Copy-Item -LiteralPath $nativeLibrary -Destination (Join-Path $nativeDir "libfluxel_android_triangle.so") -Force
$unsigned = Join-Path $out "unsigned.apk"
$aligned = Join-Path $out "aligned.apk"
$signed = Join-Path $out "01_triangle.apk"
& (Join-Path $buildTools "aapt.exe") package -f -0 so -M (Join-Path $PSScriptRoot "AndroidManifest.xml") -I $androidJar -F $unsigned $staging
if ($LASTEXITCODE -ne 0) { throw "aapt package failed." }
& (Join-Path $buildTools "zipalign.exe") -f 4 $unsigned $aligned
if ($LASTEXITCODE -ne 0) { throw "zipalign failed." }
$keystore = Join-Path $env:USERPROFILE ".android/debug.keystore"
& (Join-Path $buildTools "apksigner.bat") sign --ks $keystore --ks-pass pass:android --key-pass pass:android --out $signed $aligned
if ($LASTEXITCODE -ne 0) { throw "APK signing failed." }

adb -s $Serial install --no-incremental -r $signed
if ($LASTEXITCODE -ne 0) { throw "APK installation failed." }
adb -s $Serial logcat -c
adb -s $Serial shell am force-stop org.fluxel.triangle
adb -s $Serial shell am start -n org.fluxel.triangle/android.app.NativeActivity
if ($LASTEXITCODE -ne 0) { throw "Starting the triangle activity failed." }
Write-Host "Started 01_triangle on $Serial. Inspect with: adb -s $Serial logcat -s RustStdoutStderr:E RustStdoutStderr:W"

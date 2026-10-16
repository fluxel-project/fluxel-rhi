param(
    [ValidateSet("vulkan", "gles")][string]$Backend = "vulkan",
    [ValidateSet("01_triangle", "02_texture", "03_instancing", "04_dynamic_uniform_buffer", "05_push_constants", "06_compute", "07_offscreen", "08_msaa", "09_texture_mipmap", "10_indirect_draw", "11_occlusion_query", "12_screenshot", "13_multithreaded_recording", "14_descriptor_indexing")][string]$Demo = "01_triangle",
    [string]$Serial = "emulator-5554",
    [string]$Evidence = ""
)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "../../..")).Path
$sdk = if ($env:ANDROID_HOME) { $env:ANDROID_HOME } else { $env:ANDROID_SDK_ROOT }
if (-not $sdk) { throw "Set ANDROID_HOME or ANDROID_SDK_ROOT." }
$ndk = if ($env:ANDROID_NDK_HOME) { $env:ANDROID_NDK_HOME } else { Get-ChildItem (Join-Path $sdk "ndk") -Directory | Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty FullName }
$tools = Get-ChildItem (Join-Path $sdk "build-tools") -Directory | Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty FullName
$platform = Get-ChildItem (Join-Path $sdk "platforms") -Directory | Where-Object { Test-Path (Join-Path $_.FullName "android.jar") } | Sort-Object Name -Descending | Select-Object -First 1 -ExpandProperty FullName
$env:CARGO_TARGET_X86_64_LINUX_ANDROID_LINKER = Join-Path $ndk "toolchains/llvm/prebuilt/windows-x86_64/bin/x86_64-linux-android29-clang.cmd"
$out = Join-Path $root "target/android-rhi-capture/$Backend"
$stage = Join-Path $out "stage"
$libDir = Join-Path $stage "lib/x86_64"
New-Item -ItemType Directory -Force -Path $libDir | Out-Null
$featureList = switch ($Demo) { "02_texture" { "$Backend,demo02" } "03_instancing" { "$Backend,demo03" } "04_dynamic_uniform_buffer" { "$Backend,demo04" } "05_push_constants" { "$Backend,demo05" } "06_compute" { "$Backend,demo06" } "07_offscreen" { "$Backend,demo07" } "08_msaa" { "$Backend,demo08" } "09_texture_mipmap" { "$Backend,demo09" } "10_indirect_draw" { "$Backend,demo10" } "11_occlusion_query" { "$Backend,demo11" } "12_screenshot" { "$Backend,demo12" } "13_multithreaded_recording" { "$Backend,demo13" } "14_descriptor_indexing" { "$Backend,demo14" } default { $Backend } }
$cargoArgs = @("build", "--manifest-path", (Join-Path $PSScriptRoot "../android-runner/Cargo.toml"), "--target", "x86_64-linux-android", "--release", "--no-default-features", "--features", $featureList)
Push-Location $root
try { cargo @cargoArgs; if ($LASTEXITCODE -ne 0) { throw "Android $Backend APK library build failed." } }
finally { Pop-Location }
Copy-Item (Join-Path $PSScriptRoot "../android-runner/target/x86_64-linux-android/release/libfluxel_android_rhi_capture.so") (Join-Path $libDir "libfluxel_android_rhi_capture.so") -Force
$apk = Join-Path $out "fluxel-rhi-$Backend-$Demo.apk"
$unsigned = Join-Path $out "unsigned.apk"
& (Join-Path $tools "aapt.exe") package -f -0 so -M (Join-Path $PSScriptRoot "../android-runner/AndroidManifest.xml") -I (Join-Path $platform "android.jar") -F $unsigned $stage
if ($LASTEXITCODE -ne 0) { throw "aapt package failed." }
& (Join-Path $tools "zipalign.exe") -f 4 $unsigned (Join-Path $out "aligned.apk")
if ($LASTEXITCODE -ne 0) { throw "zipalign failed." }
$keystore = Join-Path $env:USERPROFILE ".android/debug.keystore"
& (Join-Path $tools "apksigner.bat") sign --ks $keystore --ks-pass pass:android --key-pass pass:android --out $apk (Join-Path $out "aligned.apk")
if ($LASTEXITCODE -ne 0) { throw "APK signing failed." }
adb -s $Serial install --no-incremental -r $apk
if ($LASTEXITCODE -ne 0) { throw "APK installation failed." }
adb -s $Serial logcat -c
adb -s $Serial shell am force-stop org.fluxel.rhicapture
adb -s $Serial shell am start -n org.fluxel.rhicapture/android.app.NativeActivity
Start-Sleep -Seconds 5
$logs = adb -s $Serial logcat -d -s RustStdoutStderr:I
$logs | Set-Content (Join-Path $out "logcat.txt")
if ($logs -match "RHI_(INIT|DRAW|RESIZE)_ERROR|EVENT_LOOP_ERROR") { throw "RHI reported an Android runtime error; inspect $out/logcat.txt" }
if (-not ($logs -match "RHI_FRAME_OK=2")) { throw "The Android workload did not confirm two completed frames; inspect $out/logcat.txt" }
if (-not $Evidence) { $Evidence = Join-Path $root "临时/当前验证/android/2026-10-15/native-captures/$Backend" }
New-Item -ItemType Directory -Force -Path $Evidence | Out-Null
$logs | Set-Content (Join-Path $Evidence "${Demo}_runtime.log")
if ($Demo -eq "12_screenshot") {
    $readbackLine = $logs | Where-Object { $_ -match "RHI_READBACK_OK=" } | Select-Object -First 1
    if (-not $readbackLine) { throw "The screenshot example did not confirm completed GPU readback." }
    $readbackPath = ($readbackLine -split "RHI_READBACK_OK=", 2)[1].Trim()
    adb -s $Serial pull $readbackPath (Join-Path $Evidence "${Demo}_readback.ppm")
    if ($LASTEXITCODE -ne 0) { throw "Pulling the completed GPU readback failed." }
}
adb -s $Serial shell screencap -p /sdcard/fluxel-$Backend-$Demo-01.png
adb -s $Serial pull /sdcard/fluxel-$Backend-$Demo-01.png (Join-Path $Evidence "$Demo.png") | Out-Null
Start-Sleep -Milliseconds 700
adb -s $Serial shell screencap -p /sdcard/fluxel-$Backend-$Demo-02.png
adb -s $Serial pull /sdcard/fluxel-$Backend-$Demo-02.png (Join-Path $Evidence "${Demo}_frame2.png") | Out-Null
@("backend=$Backend", "serial=$Serial", (adb -s $Serial shell getprop ro.build.version.release), (adb -s $Serial shell getprop ro.product.model), (adb -s $Serial shell getprop ro.hardware)) | Set-Content (Join-Path $Evidence "device.txt")
Write-Host "Captured RHI $Demo on $($Backend): $Evidence"

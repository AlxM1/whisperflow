#!/usr/bin/env bash
# Configure the WhisperFlow Android TWA before building.
#
# Usage:
#   ./configure.sh https://192.168.1.50:8766
#   ./configure.sh https://whisperflow.local

set -euo pipefail

if [ $# -lt 1 ]; then
    echo "Usage: $0 <server-url>"
    echo ""
    echo "Examples:"
    echo "  $0 https://192.168.1.50:8766"
    echo "  $0 https://whisperflow.local"
    exit 1
fi

SERVER_URL="$1"
HOST=$(echo "$SERVER_URL" | sed -E 's|https?://||; s|:[0-9]+$||; s|/.*||')

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
BUILD_GRADLE="$SCRIPT_DIR/app/build.gradle"

if [ ! -f "$BUILD_GRADLE" ]; then
    echo "Error: $BUILD_GRADLE not found"
    exit 1
fi

echo "Configuring Android TWA for: $SERVER_URL (host: $HOST)"

# Update hostName
sed -i "s|hostName     : \"[^\"]*\"|hostName     : \"$HOST\"|" "$BUILD_GRADLE"

# Update defaultUrl
sed -i "s|defaultUrl   : \"[^\"]*\"|defaultUrl   : \"$SERVER_URL\"|" "$BUILD_GRADLE"

# Update assetStatements site
sed -i "s|\"site\": \"[^\"]*\"|\"site\": \"$SERVER_URL\"|" "$BUILD_GRADLE"

echo "Updated $BUILD_GRADLE"
echo ""
echo "Next steps:"
echo "  1. Get your APK signing fingerprint:"
echo "     keytool -list -v -keystore ~/.android/debug.keystore -alias androiddebugkey -storepass android 2>/dev/null | grep SHA256"
echo ""
echo "  2. Set the fingerprint on your server:"
echo "     export WHISPERFLOW_APK_FINGERPRINT='<sha256-fingerprint>'"
echo ""
echo "  3. Build the APK:"
echo "     cd $SCRIPT_DIR && ./gradlew assembleDebug"
echo ""
echo "  4. Install on your device:"
echo "     adb install app/build/outputs/apk/debug/app-debug.apk"

#!/bin/bash
# Build macOS .app bundle using PyInstaller
#
# Prerequisites:
#   pip install pyinstaller
#   pip install "voice-input-method[macos]"
#
# Usage:
#   cd voice-input-method
#   bash macos/build.sh
#
# Output: dist/Rtxime.app
#
# For distribution, you'll also need to:
#   1. Sign: codesign --deep --force --sign "Developer ID Application: ..." dist/Rtxime.app
#   2. Notarize: xcrun notarytool submit dist/Rtxime.zip --apple-id ... --team-id ...
#   3. Staple: xcrun stapler staple dist/Rtxime.app

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

pyinstaller \
    --name "Rtxime" \
    --windowed \
    --onedir \
    --icon "voice_input_method/resources/icon.png" \
    --add-data "voice_input_method/resources:voice_input_method/resources" \
    --add-data "config.yaml:." \
    --add-data "hotwords.txt:." \
    --osx-bundle-identifier "com.rtxime.voiceinput" \
    --osx-entitlements-file "macos/entitlements.plist" \
    --target-arch universal2 \
    voice_input_method/__main__.py

# Replace the auto-generated Info.plist with our custom one
cp macos/Info.plist "dist/Rtxime.app/Contents/Info.plist"

echo ""
echo "Build complete: dist/Rtxime.app"
echo ""
echo "To sign for distribution:"
echo "  codesign --deep --force --sign 'Developer ID Application: YOUR_NAME' dist/Rtxime.app"
echo ""
echo "To notarize:"
echo "  ditto -c -k --keepParent dist/Rtxime.app dist/Rtxime.zip"
echo "  xcrun notarytool submit dist/Rtxime.zip --apple-id YOUR_APPLE_ID --team-id YOUR_TEAM_ID --wait"
echo "  xcrun stapler staple dist/Rtxime.app"

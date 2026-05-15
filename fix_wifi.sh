#!/bin/bash
# ============================================================
# WiFi Fix for Realtek RTL8852BE (PCI: 10ec:b520)
# HP Victus — Ubuntu 22.04 dual-boot
# ============================================================
set -e

echo ""
echo "╔══════════════════════════════════════════════════════╗"
echo "║         Vital Guardian — WiFi Fix Script            ║"
echo "╚══════════════════════════════════════════════════════╝"
echo ""

# ── Step 1: Fix the blacklist conflict ──────────────────────────────────────
echo "[1/5] Removing conflicting driver blacklists..."
# The blacklist blocks rtw_8852be but the OE driver needs to load cleanly
# Remove the blacklist so the kernel driver can bind properly
sudo rm -f /etc/modprobe.d/blacklist-rtw8852be.conf
sudo rm -f /etc/modprobe.d/blacklist-rtw88.conf
echo "  ✓ Blacklists removed"

# ── Step 2: Unload existing (broken) driver instance ────────────────────────
echo "[2/5] Unloading current driver modules..."
sudo modprobe -r rtw_8852be 2>/dev/null || true
sudo modprobe -r rtw_8852b  2>/dev/null || true
sudo modprobe -r rtw89pci   2>/dev/null || true
sudo modprobe -r rtw89core  2>/dev/null || true
sleep 1
echo "  ✓ Modules unloaded"

# ── Step 3: Force-enable the PCI device (Windows Fast Startup leaves it off) ─
echo "[3/5] Force-enabling WiFi PCI device (0000:03:00.0)..."
echo 1 | sudo tee /sys/bus/pci/devices/0000:03:00.0/enable > /dev/null
PCI_EN=$(cat /sys/bus/pci/devices/0000:03:00.0/enable)
if [ "$PCI_EN" = "1" ]; then
    echo "  ✓ PCI device enabled"
else
    echo "  ✗ PCI enable failed — will try kernel rescan"
    echo 1 | sudo tee /sys/bus/pci/rescan > /dev/null
fi

# ── Step 4: Reload the driver ───────────────────────────────────────────────
echo "[4/5] Loading WiFi driver (rtw_8852be)..."
sudo modprobe rtw89core
sudo modprobe rtw89pci
sudo modprobe rtw_8852b
sudo modprobe rtw_8852be
sleep 3

# Check if interface appeared
WLAN=$(ls /sys/class/net/ | grep -E "wlan|wlp" || true)
if [ -n "$WLAN" ]; then
    echo "  ✓ WiFi interface created: $WLAN"
else
    echo "  ✗ Interface not yet visible — checking..."
    ip link show
fi

# ── Step 5: Fix Windows Fast Startup (PERMANENT fix) ────────────────────────
echo "[5/5] Configuring permanent fixes..."

# Add kernel parameter to force PCI reset on boot
GRUB_FILE="/etc/default/grub"
CURRENT_CMDLINE=$(grep "^GRUB_CMDLINE_LINUX_DEFAULT" $GRUB_FILE | head -1)
echo "  Current GRUB line: $CURRENT_CMDLINE"

if echo "$CURRENT_CMDLINE" | grep -q "pci=realloc"; then
    echo "  ✓ pci=realloc already set"
else
    # Add pci=realloc to handle Windows Fast Startup PCIe state issues
    sudo sed -i 's/^GRUB_CMDLINE_LINUX_DEFAULT="\(.*\)"/GRUB_CMDLINE_LINUX_DEFAULT="\1 pci=realloc pcie_aspm=off"/' $GRUB_FILE
    echo "  ✓ Added pci=realloc pcie_aspm=off to GRUB"
fi

# Update GRUB
echo "  Updating GRUB..."
sudo update-grub 2>/dev/null || sudo grub-mkconfig -o /boot/grub/grub.cfg
echo "  ✓ GRUB updated"

# Update initramfs so modules load correctly on boot
echo "  Updating initramfs..."
sudo update-initramfs -u 2>/dev/null
echo "  ✓ initramfs updated"

# ── Result ───────────────────────────────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════"
echo "  Result:"
echo "══════════════════════════════════════════════════"
WLAN=$(ls /sys/class/net/ | grep -E "wlan|wlp" || true)
if [ -n "$WLAN" ]; then
    echo "  ✅ WiFi interface ACTIVE: $WLAN"
    echo ""
    echo "  Bringing up interface and scanning..."
    sudo ip link set $WLAN up 2>/dev/null || true
    sleep 2
    nmcli device status
    echo ""
    echo "  Connect via: nmcli device wifi connect 'YOUR_SSID' password 'YOUR_PASS'"
    echo "  Or use: Settings → WiFi in the desktop"
else
    echo "  ⚠ WiFi interface not yet active."
    echo ""
    echo "  ► IMPORTANT: Disable Windows Fast Startup:"
    echo "    1. Boot into Windows"
    echo "    2. Control Panel → Power Options"
    echo "    3. 'Choose what power buttons do'"
    echo "    4. UNCHECK 'Turn on fast startup'"
    echo "    5. Shut down Windows (not restart)"
    echo "    6. Boot back into Ubuntu"
    echo ""
    echo "  After disabling Fast Startup, run: bash fix_wifi.sh again"
    echo ""
    echo "  Alternative right now: try a full reboot with:"
    echo "    sudo reboot"
fi
echo ""
echo "  Kernel messages about WiFi:"
sudo dmesg | grep -iE "rtw|8852|wlan" | tail -15

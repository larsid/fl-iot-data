#!/bin/sh
set -e

for attempt in 1 2 3 4 5 6 7 8 9 10; do
    found=0
    for iface in $(ls /sys/class/net 2>/dev/null); do
        case "$iface" in
            lo|eth0) continue ;;
        esac
        ip link set "$iface" up 2>/dev/null && found=1
    done
    [ "$found" = "1" ] && break
    sleep 1
done

exec "$@"

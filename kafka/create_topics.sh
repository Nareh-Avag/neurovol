#!/bin/bash
# create_topics.sh
# Creates Kafka topics required by the Neural Volatility Surface Forecasting System.
#
# Usage:
#   KAFKA_HOST=broker:9092 UNDERLYING=SPX ./create_topics.sh
#
# Environment variables:
#   KAFKA_HOST   – Kafka bootstrap server (default: localhost:9092)
#   UNDERLYING   – Ticker symbol for the underlying (default: SPX)
#   PARTITIONS   – Number of partitions per topic (default: 12)
#   REPLICATION  – Replication factor (default: 1)

set -euo pipefail

KAFKA_HOST=${KAFKA_HOST:-localhost:9092}
UNDERLYING=${UNDERLYING:-SPX}
PARTITIONS=${PARTITIONS:-12}
REPLICATION=${REPLICATION:-1}

topics=(
    "options.raw.${UNDERLYING}"
    "options.clean.${UNDERLYING}"
    "options.iv.${UNDERLYING}"
    "surface.snapshots.${UNDERLYING}"
    "signals.vol.${UNDERLYING}"
)

echo "============================================================"
echo " Neural Volatility – Kafka Topic Bootstrap"
echo " Broker    : ${KAFKA_HOST}"
echo " Underlying: ${UNDERLYING}"
echo " Partitions: ${PARTITIONS}  Replication: ${REPLICATION}"
echo "============================================================"

for topic in "${topics[@]}"; do
    kafka-topics.sh --bootstrap-server "$KAFKA_HOST" \
        --create --if-not-exists \
        --topic "$topic" \
        --partitions "$PARTITIONS" \
        --replication-factor "$REPLICATION"
    echo "Created topic: $topic"
done

echo "------------------------------------------------------------"
echo "All topics created."
echo ""
echo "Listing topics on ${KAFKA_HOST}:"
kafka-topics.sh --bootstrap-server "$KAFKA_HOST" --list | grep -E "options\.|surface\.|signals\."

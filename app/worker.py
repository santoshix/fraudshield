"""Kafka consumer: an example of independent, asynchronous transaction processing."""
import json
import logging
import os
import time
from kafka import KafkaConsumer
from app.db import init_db, SessionLocal
from app.main import save_transaction
from app.schemas import TransactionInput

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
BROKERS = os.getenv("KAFKA_BOOTSTRAP_SERVERS", "kafka:29092").split(",")

def main():
    init_db()
    while True:
        try:
            logging.info("Connecting Kafka worker to %s", BROKERS)
            consumer = KafkaConsumer(
                "fraud.incoming", bootstrap_servers=BROKERS,
                group_id="fraud-risk-worker", auto_offset_reset="earliest",
                enable_auto_commit=False,
                value_deserializer=lambda x: json.loads(x.decode("utf-8")),
                consumer_timeout_ms=1000,
            )
            logging.info("Kafka consumer online")
            while True:
                batches = consumer.poll(timeout_ms=1000, max_records=50)
                for _, messages in batches.items():
                    for message in messages:
                        with SessionLocal() as session:
                            try:
                                data = TransactionInput.model_validate(message.value)
                                scored = save_transaction(data, session, "KAFKA")
                                logging.info("Scored %s: %.2f (%s)", scored["transaction_id"], scored["risk_score"], scored["status"])
                                consumer.commit()  # commit only on successful DB write
                            except Exception:
                                session.rollback()
                                logging.exception("Kafka message failed; not committing its offset")
                                raise
        except Exception:
            logging.exception("Kafka/database unavailable; retrying in 5 seconds")
            time.sleep(5)

if __name__ == "__main__":
    main()

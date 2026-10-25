"""Count the messages in each SynAudit topic and print one sample message per topic.

    python quality/check_topics.py --bootstrap localhost:29092
"""
import argparse
import json

TOPICS = ["adult-real", "adult-synthetic", "credit-default-real", "credit-default-synthetic"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bootstrap", default="localhost:29092")
    ap.add_argument("--topics", nargs="*", default=TOPICS)
    args = ap.parse_args()

    from kafka import KafkaConsumer, TopicPartition

    consumer = KafkaConsumer(bootstrap_servers=args.bootstrap, consumer_timeout_ms=3000,
                             auto_offset_reset="earliest", enable_auto_commit=False)
    print(f"{'topic':32} {'messages':>9}")
    for topic in args.topics:
        parts = consumer.partitions_for_topic(topic)
        if not parts:
            print(f"{topic:32} {'(missing)':>9}")
            continue
        tps = [TopicPartition(topic, p) for p in parts]
        begin, end = consumer.beginning_offsets(tps), consumer.end_offsets(tps)
        total = sum(end[tp] - begin[tp] for tp in tps)
        print(f"{topic:32} {total:>9}")
        if total:
            consumer.assign(tps[:1])
            consumer.seek_to_beginning(tps[0])
            for msg in consumer:
                try:
                    shown = json.dumps(json.loads(msg.value), ensure_ascii=False)
                except Exception:
                    shown = msg.value[:300]
                print(f"    sample: {shown[:300]}")
                break
    consumer.close()


if __name__ == "__main__":
    main()

from confluent_kafka.admin import AdminClient, NewTopic

from . import config


def ensure_topics(partitions: int = 3) -> None:
    admin = AdminClient({"bootstrap.servers": config.KAFKA_BOOTSTRAP})
    existing = set(admin.list_topics(timeout=10).topics)
    new = [NewTopic(t, num_partitions=partitions, replication_factor=1)
           for t in (config.TOPIC_APPLICATIONS, config.TOPIC_REVIEW) if t not in existing]
    if not new:
        return
    for t, f in admin.create_topics(new).items():
        f.result()
        print(f"created topic {t}")

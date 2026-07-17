from datasets import load_dataset

dataset = load_dataset("rajpurkar/squad")
sample = dataset["train"].select(range(2000))

topics = ["Antibiotics", "Beyoncé", "Frédéric_Chopin", "Genocide", "Montana", "University_of_Notre_Dame"]

for topic in topics:
    print(f"\n{'='*60}")
    print(f"TOPIC: {topic}")
    print('='*60)

    # Get up to 5 example questions already written for this topic
    count = 0
    for example in sample:
        if example["title"] == topic:
            print(f"- {example['question']}")
            count += 1
        if count >= 5:
            break
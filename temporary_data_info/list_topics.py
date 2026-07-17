from datasets import load_dataset

dataset = load_dataset("rajpurkar/squad")
sample = dataset["train"].select(range(2000))

seen_titles = set()
for example in sample:
    seen_titles.add(example["title"])

print(f"Total unique topics in first 2000 examples: {len(seen_titles)}\n")
for title in sorted(seen_titles):
    print(title)
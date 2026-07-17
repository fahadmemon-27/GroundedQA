from nli_scorer import nli_model, LABELS

# Hand-labeled test cases: (evidence, claim, TRUE_label)
# TRUE_label is OUR human judgment of the correct answer
test_cases = [
    # Clear supported cases
    ("The Grotto is a Marian place of prayer and reflection.",
     "The Grotto is a Marian place of prayer and reflection.",
     "supported"),

    ("Atop the Main Building's gold dome is a golden statue of the Virgin Mary.",
     "There is a golden statue on top of the Main Building.",
     "supported"),

    ("The iPod is a line of portable media players developed and marketed by Apple Inc.",
     "Apple makes the iPod.",
     "supported"),

    # Clear contradicted cases
    ("The match was played at the Gabba due to COVID restrictions.",
     "The match was held in Adelaide.",
     "contradicted"),

    ("Frédéric François Chopin was a Polish and French composer.",
     "Chopin was only German.",
     "contradicted"),

    # Clear unclear cases (evidence doesn't mention this at all)
    ("Atop the Main Building's gold dome is a golden statue of the Virgin Mary.",
     "The Main Building was built in 1900.",
     "unclear"),

    ("The Grotto is a Marian place of prayer and reflection.",
     "The Grotto is visited by thousands of tourists every year.",
     "unclear"),

    # Known tricky cases (from our earlier testing - these are the ones
    # we know NLI might get wrong)
    ("There are multitudes of religious statues and artwork around campus, most prominent of which are the statue of Mary on the Main Building, the Notre Dame Grotto, and the Word of Life mural.",
     "The statue of Mary on the Main Building is the Notre Dame Grotto.",
     "contradicted"),  # our human judgment: this is false equivalence, should be flagged wrong

    ("The Grotto at Notre Dame is a Marian place of prayer and reflection.",
     "The Grotto at Notre Dame is a Marian place of prayer and reflection.",
     "supported"),

    ("Beyoncé started becoming popular in the late 1990s as a member of Destiny's Child.",
     "Beyoncé started becoming popular in the 1990s.",
     "supported"),
]

# Run NLI on all test cases
predicted_labels = []
true_labels = []

print("=== Individual Results ===\n")
for evidence, claim, true_label in test_cases:
    score = nli_model.predict([(evidence, claim)])
    predicted_label = LABELS[score[0].argmax()]

    predicted_labels.append(predicted_label)
    true_labels.append(true_label)

    match = "✅" if predicted_label == true_label else "❌"
    print(f"{match} True: {true_label:<12} | Predicted: {predicted_label:<12} | Claim: {claim}")

# Calculate simple accuracy
correct = sum(1 for p, t in zip(predicted_labels, true_labels) if p == t)
accuracy = (correct / len(test_cases)) * 100

print(f"\n=== Overall Accuracy: {accuracy:.1f}% ({correct}/{len(test_cases)}) ===\n")

# Build confusion matrix manually
print("=== Confusion Matrix ===")
print("(rows = true label, columns = predicted label)\n")

label_options = ["contradicted", "supported", "unclear"]
print(f"{'':<15}", end="")
for pred_label in label_options:
    print(f"{pred_label:<15}", end="")
print()

for true_label in label_options:
    print(f"{true_label:<15}", end="")
    for pred_label in label_options:
        count = sum(1 for p, t in zip(predicted_labels, true_labels) if t == true_label and p == pred_label)
        print(f"{count:<15}", end="")
    print()
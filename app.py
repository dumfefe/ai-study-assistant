
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load the pretrained AI model
model_name = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)


def study_assistant():
    print("=== AI Study Assistant ===")
    print("Ask a question or enter a topic.")
    print("Type 'exit' to stop.\n")

    while True:
        question = input("You: ")

        if question.lower() == "exit":
            print("Goodbye!")
            break

        prompt = (
            "Answer the following question clearly and simply "
            "for a beginner student:\n\n"
            + question
        )

        inputs = tokenizer(
            prompt,
            return_tensors="pt"
        )

        outputs = model.generate(
            **inputs,
            max_new_tokens=150
        )

        answer = tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )

        print("\nAI:", answer)
        print()


if __name__ == "__main__":
    study_assistant()

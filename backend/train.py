from bpe_tokenizer import BPETokenizer
from pre_tokenizer import PreTokenizer, ProcessPoolTokenizer

def main():
    with open("datasets/shakespeare.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = PreTokenizer() #ProcessPoolTokenizer()

    print("Training tokenizer...")
    tokenizer.train(text, vocab_size=13000)
    print("Training complete.")

    tokenizer.save("models/shakespeare_10000.pkl")
    # tokenizer.load("models/shakespeare_13000.pkl")

    sentence = "Oliver was"
    ids = tokenizer.encode(sentence)

    print("Encoded:", ids)

    decoded = tokenizer.decode(ids)
    print("Decoded:", decoded)

    print("Compression:", tokenizer.compression_ratio(sentence))
    # print("\nFirst 5 merges:")
    # for h in tokenizer.history[:5]:
    #     print(h)

if __name__ == "__main__":
    main()
import json
import pickle
from utils.vocab import Vocabulary

if __name__ == "__main__":
    with open('data/cleaned_captions.json', 'r') as f:
        captions_data = json.load(f)

    all_captions = []
    for cap_list in captions_data.values():
        all_captions.extend(cap_list)

    vocab = Vocabulary(freq_threshold=5)
    vocab.build_vocabulary(all_captions)

    with open('data/vocab.pkl', 'wb') as f:
        pickle.dump(vocab, f)

    print(f"✅ Vocabulary saved to data/vocab.pkl with {len(vocab)} tokens.")

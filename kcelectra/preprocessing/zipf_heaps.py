import matplotlib.pyplot as plt
from collections import Counter

def plot_zipf(all_words):
    word_counts_all = Counter(all_words)
    sorted_word_counts = sorted(word_counts_all.values(), reverse=True)

    plt.figure(figsize=(8, 6))
    plt.loglog(range(1, len(sorted_word_counts) + 1), sorted_word_counts)
    plt.title("Zipf의 법칙 - 단어 순위 vs 빈도")
    plt.xlabel("단어 순위 (Rank)")
    plt.ylabel("단어 빈도 (Frequency)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_heaps(tokenized_comments):
    total_tokens = 0
    unique_tokens = set()
    total_token_counts = []
    unique_token_counts = []
    
    for tokens in tokenized_comments:
        for token in tokens:
            total_tokens += 1
            unique_tokens.add(token)
            total_token_counts.append(total_tokens)
            unique_token_counts.append(len(unique_tokens))

    plt.figure(figsize=(8, 6))
    plt.plot(total_token_counts, unique_token_counts)
    plt.title("Heaps의 법칙 - 전체 단어 수 vs 고유 단어 수")
    plt.xlabel("전체 단어 수 (Total Tokens)")
    plt.ylabel("고유 단어 수 (Unique Tokens)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
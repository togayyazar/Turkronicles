class TrieNode:

    def __init__(self):
        self.children = {}
        self.is_end_of_word = False
        self.frequency = 0
        self.prefix_count = 0



class Trie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str):
        self.insert_with_frequency(word, 1)

    def insert_with_frequency(self, word: str, freq: int):
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
            node.prefix_count += freq
        node.is_end_of_word = True
        node.frequency += freq

    def search(self, word: str) -> int:
        node = self.root
        for char in word:
            if char not in node.children:
                return 0
            node = node.children[char]
        return node.frequency if node.is_end_of_word else 0

    def starts_with(self, prefix: str) -> int:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return 0
            node = node.children[char]
        return node.prefix_count

    def get_words_with_prefix(self, prefix: str):

        def dfs(node, prefix, words):
            if node.is_end_of_word:
                words.append((prefix, node.frequency))
            for char, child_node in node.children.items():
                dfs(child_node, prefix + char, words)

        node = self.root
        for char in prefix:
            if char not in node.children:
                return []
            node = node.children[char]

        words = []
        dfs(node, prefix, words)
        return words

    def total_words_count(self) -> int:
        def dfs(node):
            total = node.frequency if node.is_end_of_word else 0
            for child in node.children.values():
                total += dfs(child)
            return total

        return dfs(self.root)

    def get_all_words(self):
        words = list(zip(*self.get_words_with_prefix("")))[0]
        return words

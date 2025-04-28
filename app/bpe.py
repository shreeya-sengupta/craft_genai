# import
import itertools
from collections import Counter
from collections import defaultdict


final_vocab = ['Walker </w>', 'w alke d </w>', 'a </w>', 'l o n g </w>', 'w alk </w>', 'a t </w>', 's o m e u n k n o w n b e a c h </w>']
learned_merges = [('a', 'l'), ('al', 'k'), ('alk', 'e'), ('W', 'alke'), ('Walke', 'r')]

def apply_bpe(test_word, merges):
    test_tokens = list(test_word) + ['</w>']
    i = 0
    while i < len(test_tokens) - 1:
        pair = (test_tokens[i], test_tokens[i+1])
        if pair in reversed(merges):
            test_tokens[i] = ''.join(pair)
            del test_tokens[i+1]
            i = max(i-1, 0)  # re-check merged token with previous one
        else:
            i += 1
    return test_tokens

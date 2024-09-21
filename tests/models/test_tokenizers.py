from unittest import TestCase
from stenographer.models.tokenizers impoer SyllableTokenizer


class SyllableTokenizerTest(TestCase):
    def setup(self):
        self.token_file_path = 'resources/tests/syllable_tokens.json'

    def test_decode_function(self):
        tokenizer = SyllableTokenizerTest.from_file()
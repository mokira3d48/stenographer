from unittest import TestCase
from stenographer.models.tokenizers import PhoneticTokenizer


class PhoneticTokenizerTest(TestCase):
    def setup(self):
        self.token_file_path = 'resources/tests/syllable_tokens.json'
        self.abbr_fp = 'resources/fr/abbrevs.json'
        self.expr_pron_fp = 'resources/fr/expr_pron.json'
        self.word_pron_fp = 'resources/fr/word_pron.json'
        self.phonemes_fp = 'resources/fr/phonemes_vocab.json'

    def test_decode_function(self):
        tokenizer = PhoneticTokenizer.get_instance(self.abbr_fp,
                                                   self.expr_pron_fp,
                                                   self.word_pron_fp,
                                                   self.phonemes_fp)
        text = """
        L’idée remonte aux années 1950 avec John von Neumann et a été débattue
        par I. J. Good, V. Vinge (4) ou  R. Kurzweil (5). On la retrouve
        fréquemment dans la culture populaire avec 2001 Odyssée de l’espace,
        Terminator, Minority Report, Matrix, et plus récemment Her ou encore
        Transcendance.
        """
        tokenizer([text])

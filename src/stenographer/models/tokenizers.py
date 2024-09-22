import os
import re
import logging
import json

import nltk
from num2words import num2words


_LOG = logging.getLogger(__name__)
_NUMBERS = set('0123456789')
_LETTERS = set('abcdefghijklmnopqrstuvwxyz')


def _read_json_file(file_path):
    """Function to read a json file
    """
    with open(file_path, mode='r', encoding='utf-8') as f:
        json_content = json.load(f)
        return json_content


class _Transformer:
    def transform(self, x):
        raise NotImplementedError

    def __call__(self, inp):
        if isinstance(inp, list):
            out = []
            for x in inp:
                res = self.transform(x)
                out.append(res)
            return out
        else:
            res = self.transform(inp)
            return res


class _Text2Lower(_Transformer):
    """Convert text to lower characters
    """
    def __init__(self):
        super().__init__()

    def transform(self, x):
        assert x is not None, (
            "The text which we want to convert into lower is not defined.")
        return x.lower()


class _Abbreviation(_Transformer):
    """Function of abbreviation transformation

    :arg abbr_dict:
        The dictionary of abbreviation mapped to the words
        or expressions existing in to language vocabulary.
    :type abbr_dict: `dict`
    """
    def __init__(self, abbr_dict):
        super().__init__()
        self.abbr_dict = abbr_dict

    def transform(self, x):
        # If element is existing then we return its equivalent,
        # otherwise, we return keep empty word.
        x_split = x.split()
        res = [''] * len(x_split)
        for idx, word in enumerate(x_split):
            if word in self.abbr_dict:
                res[idx] = self.abbr_dict[word]
        return res


class _PuncTokenization(_Transformer):
    """
    Perform a tokenization using the punctuation contained in text.
    """
    def __init__(self):
        super().__init__()

    def transform(self, x):
        assert x is not None, (
            "The text which we want to tokenize is not defined.")
        return nltk.wordpunct_tokenize(x)


class _NumberTokenizer(_Transformer):
    """
    Separate the numbers from the text, for example:
    "123ème" is a case which is not handle by the punctuation tokenizer.
    So we will try to separate some number from the words.

    :param letters:
        The alphabet of all the letter contained in the language.
    :param numbers:
        The alphabet of all the number contained in the language.

    :type letters: `list`
    :type numbers: `list`
    """
    def __init__(self, numbers, letters):
        super().__init__()
        self.numbers = numbers
        self.letters = letters

    @staticmethod
    def _filters(string, pattern):
        word_split = re.split(rf"[{''.join(pattern)}]", string)
        result = list(filter(lambda x: bool(x), word_split))
        return result

    @staticmethod
    def _pos_dict(words, string):
        """
        Returns mapping between each word
        and its position in string.
        """
        out = {}
        for w in words:
            out[w] = string.index(w)
        return out

    def transform(self, x):
        assert x is not None, (
            "The text in which we want to separate the numbers"
            "from the words is not defined.")
        x_split = x.split()
        out = []
        for word in x_split:
            number_found = any(map((lambda c: c in self.numbers), word))
            letter_found = any(map((lambda c: c in self.letters), word))
            if not (number_found and letter_found):
                out.append(word)
                continue

            # We will extract the number and letter from the word:
            numbers = self._filters(word, self.numbers)
            letters = self._filters(word, self.letters)

            # Build position dictionaries:
            # Example: {"123": 0, "ème": 3}
            numbers_pd = self._pos_dict(numbers, word)
            letters_pd = self._pos_dict(letters, word)
            merge_pd = {**numbers_pd, **letters_pd}
            # TODO: place each word at corresponding position.


class _Num2Text(_Transformer):
    """Convert a number to its equivalent text

    :param lang:
        The language selected.
    :param remove_dash:
         If True, we remove the dash character after transcription
         otherwise, we do nothing, after transcription.

    :type lang: `str`
    :type remove_dash: `bool`
    """
    _PATTERN = r'\b\d+(?:[,.]\d+)?\b'

    def __init__(self, lang='fr', remove_dash=False):
        self.lang = lang
        self.remove_dash = remove_dash

    def to_text(self, num):
        assert num is not None, "The number should not be none."
        if not isinstance(num, int) and not isinstance(num, float):
            raise TypeError("The number must be a int of float type.")
        return num2words(num, lang=self.lang)

    def transform(self, x):
        assert x is not None, 'The text should not be none.'
        if not isinstance(x, str):
            raise TypeError("The type must be a string.")
        if not x:
            return ''
        result = x
        num_list = re.findall(self._PATTERN, x)
        for str_num in num_list:
            num = str_num
            if ',' in num:
                num = num.replace(',', '.')  # eg: 3.1415 -> 3,1415
            if '.' in num:
                value = float(num)
            else:
                value = int(num)
            trans = self.to_text(value)
            if self.remove_dash:
                trans = trans.replace('-', ' ')
            result = result.replace(str_num, trans, 1)
        return result


class _Prononciation(_Transformer):
    """
    Transformation of text into prononciation using IPA dictionary.

    :param ipa:
        The IPA dictionary that will be used
        to transform text into prononciation.
    :type ipa: `dict`
    """
    def __init__(self, ipa):
        super().__init__()
        self.ipa = ipa

    def transform(self, x):
        assert x is not None, (
            "The text which we want to transform"
            "into prononciation is not defined.")
        text_split = x.split()
        results = []
        for word in text_split:
            pron = self.ipa.get(word, '')
            results.append(pron)
        return results


class PhoneticTokenizer(_Transformer):
    """Phonetic tokenization

    :arg phonemes_vocab:
        The all phoneme value vocab.
    :arg abbr_transform:
        Transform the abbreviation into plain text.
    :arg punc_tokenizer:
        Tokenize the text into sequence according
        punctuations contained into the text.
    :arg num_transcript:
        Transcript the number contained in text
        into the letter text.
    :arg expr_transform:
        Convert the expressions found into its prononciation.
    :arg word_transform:
        Convert the words found of text into its prononciation.

    :type phonemes_vocab: `list`
    :type abbr_transform: `_Transformer`
    :type punc_tokenizer: `_Transformer`
    :type num_transcript: `_Transformer`
    :type expr_transform: `_Transformer`
    :type word_transform: `_Transformer`
    """
    def __init__(
            self,
            phonemes_vocab,
            abbr_transform,
            punc_tokenizer,
            num_transcript,
            expr_transform,
            word_transform,
    ):
        self.phonemes_vocab = phonemes_vocab
        self.abbr_transform = abbr_transform
        self.punc_tokenizer = punc_tokenizer
        self.num_transcript = num_transcript
        self.expr_transform = expr_transform
        self.word_transform = word_transform

    @classmethod
    def get_instance(
            cls,
            abbr_dict,
            expr_pron_dict,
            word_pron_dict,
            phonemes_vocab,
    ):
        """
        Class method to build and return an instance of the phonetic tokenizer.

        :param abbr_dict:
            The JSON file path of the abbreviation dictionary.
        :param expr_pron_dict:
            The JSON file path of expression prononciations dictionary.
        :param word_pron_dict:
            The JSON file path of the word prononciations dictionary.
        :param phonemes_vocab:
            The JSON file path of the phoneme vocabularies.

        :rtype: `PhoneticTokenizer`
        """
        for file_path in [abbr_dict, expr_pron_dict,
                          word_pron_dict, phonemes_vocab]:
            assert file_path is not None, (
                "A value of an argument is not defined.")
            if not os.path.isfile(file_path):
                raise FileNotFoundError(f"No such file at {file_path}.")

        # Load the phoneme vocab:
        phon_dict = _read_json_file(phonemes_vocab)
        phon_list = list(phon_dict.keys())
        phon_vocab = sorted(phon_list)

        # Given all instance that we need hase same constructor signature
        # so, we mapp each class of the transformer with each json file path.
        # And then, we will load each json file path and instantiate
        # the corresponding transformer class.
        mapping = {abbr_dict:_Abbreviation,
                   expr_pron_dict: _Prononciation,
                   word_pron_dict: _Prononciation}
        transformer_instances = []
        for file_path, transformer_class in mapping.items():
            dict_loaded = _read_json_file(file_path)
            instance = transformer_class(file_path)
            transformer_instances.append(instance)
        abbreviation, expr_transform, word_transform = transformer_instances

        # Now, we will instantiate the reste of dependence:
        punc_tokenizer = _PuncTokenization()
        num_transform = _Num2Text(lang='fr')  # `remove_dash` will be adjusted;

        # Instantiate the tokenizer:
        new_instance = cls(phon_vocab, abbreviation, punc_tokenizer,
                           num_transform, expr_transform, word_transform)
        return new_instance

    def encode(self, x):
        assert x is not None, (
            "The text value which will be tokenized is not defined.")
        x = x.lower()
        x = self.abbr_transform(x)


    def transform(self, x):
        return self.encode(x)

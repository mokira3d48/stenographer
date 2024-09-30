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
        if not isinstance(inp, list):
            raise TypeError("The input must be a list type.")
        out = []
        for x in inp:
            res = self.transform(x)
            out.append(res)
        return out


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
            else:
                res[idx] = word
        return ' '.join(res)


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
        indexes = []
        wdict = {}
        for w in words:
            # This sub-word `w` can be found at several places.
            matches = re.finditer(rf'{re.escape(w)}', string)
            for match in matches:
                # _LOG.debug(str(match.start()))
                ind = match.start()
                indexes.append(ind)
                wdict[ind] = w
        out = sorted(indexes)
        return out, wdict

    def transform(self, x):
        assert x is not None, (
            "The text in which we want to separate the numbers"
            "from the words is not defined.")
        # x_split = x.split()
        out = []
        for word in x:
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
            num_pos, num_dict = self._pos_dict(numbers, word)
            wrd_pos, wrd_dict = self._pos_dict(letters, word)
            # _LOG.debug(str(num_pos))
            # _LOG.debug(str(num_dict))
            wrd_res = {**num_dict, **wrd_dict}
            pos_res = num_pos + wrd_pos
            pos_res = sorted(pos_res)
            for pos in pos_res:
                out.append(wrd_res[pos])

        return out


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

    def __init__(self, numbers, lang='fr', remove_dash=False):
        self.numbers = numbers
        self.lang = lang
        self.remove_dash = remove_dash

    def to_text(self, num):
        assert num is not None, "The number should not be none."
        if not isinstance(num, int) and not isinstance(num, float):
            raise TypeError("The number must be a int of float type.")
        return num2words(num, lang=self.lang)

    def transform(self, x):
        assert x is not None, 'The text should not be none.'
        if not isinstance(x, list):
            raise TypeError("The type must be a list.")
        if not x:
            return ''
        result = x[:]
        # num_list = re.findall(self._PATTERN, x)
        for ind, num in enumerate(x):
            number_found = all(map((lambda c: c in self.numbers), num))
            if not number_found:
                continue
            if ',' in num:
                num = num.replace(',', '.')  # eg: 3.1415 -> 3,1415
            if '.' in num:
                value = float(num)
            else:
                value = int(num)
            trans = self.to_text(value)
            if self.remove_dash:
                trans = trans.replace('-', ' ')
            # result = result.replace(str_num, trans, 1)
            result[ind] = trans
        return result


class _ExprPronunciation(_Transformer):
    """
    Transformation of expressions into their pronunciation
    using IPA dictionary.

    :param ipa:
        The IPA dictionary that will be used
        to transform text into pronunciation.
    :type ipa: `dict`
    """
    def __init__(self, ipa):
        super().__init__()
        self.ipa = ipa

    def transform(self, x):
        assert x is not None, (
            "The text which we want to transform"
            "into pronunciation is not defined.")
        pron_found = []
        x_string = ' '.join(x)
        for expression, pronunciation in self.ipa.items():
            pattern = rf"{expression}"
            # _LOG.debug(pattern)
            # occurrences_found = re.match(pattern, x_string)
            occurrences_found = pattern in x_string
            # _LOG.debug(str(occurrences_found))
            if not occurrences_found:
                continue
            x_string = x_string.replace(expression, pronunciation)
            pron_found.append(pronunciation)

        out = x_string.split()
        return out, pron_found


class _WordPronunciation(_Transformer):
    """
    Transformation of words into their pronunciation using IPA dictionary.

    :param ipa:
        The IPA dictionary that will be used
        to transform text into pronunciation.
    :type ipa: `dict`
    """
    def __init__(self, ipa):
        super().__init__()
        self.ipa = ipa

    def transform(self, x):
        assert x is not None, (
            "The text which we want to transform"
            "into pronunciation is not defined.")
        # text_split = x.split()
        results = []
        for word in x:
            if not word:
                results.append(word)
                continue
            pron = self.ipa.get(word, word)
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
        Convert the expressions found into its pronunciation.
    :arg word_transform:
        Convert the words found of text into its pronunciation.

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
            num_tokenizer,
            num_transcript,
            expr_transform,
            word_transform,
    ):
        self.phonemes_vocab = phonemes_vocab
        self.abbr_transform = abbr_transform
        self.punc_tokenizer = punc_tokenizer
        self.num_tokenizer = num_tokenizer
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
            The JSON file path of expression pronunciations dictionary.
        :param word_pron_dict:
            The JSON file path of the word pronunciations dictionary.
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
        # so, we map each class of the transformer with each json file path.
        # And then, we will load each json file path and instantiate
        # the corresponding transformer class.
        mapping = [(abbr_dict, _Abbreviation),
                   (expr_pron_dict, _ExprPronunciation),
                   (word_pron_dict, _WordPronunciation)]
        transformer_instances = []
        for file_path, transformer_class in mapping:
            dict_loaded = _read_json_file(file_path)
            instance = transformer_class(dict_loaded)
            transformer_instances.append(instance)
        abbreviation, expr_transform, word_transform = transformer_instances

        # Now, we will instantiate the rest of dependence:
        punc_tokenizer = _PuncTokenization()
        num_tokenizer = _NumberTokenizer(list(_NUMBERS), list(_LETTERS))
        num_transform = _Num2Text(_NUMBERS, lang='fr')
        # `remove_dash` will be adjusted;

        # Instantiate the tokenizer:
        new_instance = cls(phon_vocab, abbreviation, punc_tokenizer,
                           num_tokenizer, num_transform, expr_transform,
                           word_transform)
        return new_instance

    @staticmethod
    def _split(seq):
        res = []
        for elem in seq:
            res.extend(elem.split())
        return res

    @staticmethod
    def _set_to_blank(seq, words):
        res = seq[:]
        ret = {}
        for i, w in enumerate(seq):
            if w not in words:
                continue
            res[i] = ''
            ret[w] = i
        return res, ret

    @staticmethod
    def _fill(seq, word):
        for w, i in word.items():
            seq[i] = w
        return seq

    def encode(self, x):
        assert x is not None, (
            "The text value which will be tokenized is not defined.")
        x = [s.lower() for s in x]
        # x = [s.split() for s in x]
        x = self.abbr_transform(x)
        x = self.punc_tokenizer(x)
        x = self.num_tokenizer(x)
        x = self.num_transcript(x)

        x = [self._split(s) for s in x]
        x = self.expr_transform(x)

        x_cpy = []
        r_dict = []
        for s, pho in x:
            s_, r_ = self._set_to_blank(s, pho)
            x_cpy.append(s_)
            r_dict.append(r_)
        x_cpy = self.word_transform(x_cpy)
        x = [self._fill(s, r_) for s, r_ in zip(x_cpy, r_dict)]

        _LOG.debug("RESULTS: " + str(x))
        return x

    def transform(self, x):
        return self.encode(x)

    def __call__(self, inp):
        return self.transform(inp)

# -*- coding: utf-8 -*-
"""Unit tests for the LEKCut Thai word tokenization library."""
import unittest

from lekcut import word_tokenize


class TestWordTokenizeDeepcut(unittest.TestCase):
    """Tests for the default deepcut model."""

    def test_basic_tokenization(self):
        result = word_tokenize("ทดสอบการตัดคำ", model="deepcut")
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)
        self.assertEqual("".join(result), "ทดสอบการตัดคำ")

    def test_known_output(self):
        result = word_tokenize("ทดสอบการตัดคำ", model="deepcut")
        self.assertEqual(result, ["ทดสอบ", "การ", "ตัด", "คำ"])

    def test_empty_string(self):
        result = word_tokenize("", model="deepcut")
        self.assertEqual(result, [])

    def test_single_word(self):
        result = word_tokenize("สวัสดี", model="deepcut")
        self.assertIsInstance(result, list)
        self.assertEqual("".join(result), "สวัสดี")

    def test_with_spaces(self):
        result = word_tokenize("สวัสดี ครับ", model="deepcut")
        self.assertIsInstance(result, list)
        self.assertEqual("".join(result), "สวัสดี ครับ")

    def test_default_model(self):
        """word_tokenize defaults to deepcut."""
        result = word_tokenize("ทดสอบการตัดคำ")
        self.assertEqual(result, ["ทดสอบ", "การ", "ตัด", "คำ"])


class TestWordTokenizeAttacutSC(unittest.TestCase):
    """Tests for the attacut-sc model."""

    def test_basic_tokenization(self):
        result = word_tokenize("ทดสอบการตัดคำ", model="attacut-sc")
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)
        self.assertEqual("".join(result), "ทดสอบการตัดคำ")

    def test_empty_string(self):
        result = word_tokenize("", model="attacut-sc")
        self.assertEqual(result, [])

    def test_output_joins_to_input(self):
        text = "ภาษาไทยสวยงาม"
        result = word_tokenize(text, model="attacut-sc")
        self.assertEqual("".join(result), text)


class TestWordTokenizeAttacutC(unittest.TestCase):
    """Tests for the attacut-c model."""

    def test_basic_tokenization(self):
        result = word_tokenize("ทดสอบการตัดคำ", model="attacut-c")
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)
        self.assertEqual("".join(result), "ทดสอบการตัดคำ")

    def test_empty_string(self):
        result = word_tokenize("", model="attacut-c")
        self.assertEqual(result, [])

    def test_output_joins_to_input(self):
        text = "ภาษาไทยสวยงาม"
        result = word_tokenize(text, model="attacut-c")
        self.assertEqual("".join(result), text)


class TestWordTokenizeOskut(unittest.TestCase):
    """Tests for the oskut model."""

    def test_basic_tokenization(self):
        result = word_tokenize("ทดสอบการตัดคำ", model="oskut")
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)
        self.assertEqual("".join(result), "ทดสอบการตัดคำ")

    def test_empty_string(self):
        result = word_tokenize("", model="oskut")
        self.assertIsInstance(result, list)
        self.assertEqual(result, [])

    def test_output_joins_to_input(self):
        text = "ภาษาไทยสวยงาม"
        result = word_tokenize(text, model="oskut")
        self.assertEqual("".join(result), text)


class TestWordTokenizeSefrWs1000(unittest.TestCase):
    """Tests for the sefr-ws1000 model."""

    def test_basic_tokenization(self):
        result = word_tokenize("ทดสอบการตัดคำ", model="sefr-ws1000")
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)
        self.assertEqual("".join(result), "ทดสอบการตัดคำ")

    def test_empty_string(self):
        result = word_tokenize("", model="sefr-ws1000")
        self.assertEqual(result, [])

    def test_output_joins_to_input(self):
        text = "ภาษาไทยสวยงาม"
        result = word_tokenize(text, model="sefr-ws1000")
        self.assertEqual("".join(result), text)


class TestWordTokenizeSefrTnhc(unittest.TestCase):
    """Tests for the sefr-tnhc model."""

    def test_basic_tokenization(self):
        result = word_tokenize("ทดสอบการตัดคำ", model="sefr-tnhc")
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)
        self.assertEqual("".join(result), "ทดสอบการตัดคำ")

    def test_empty_string(self):
        result = word_tokenize("", model="sefr-tnhc")
        self.assertEqual(result, [])

    def test_output_joins_to_input(self):
        text = "ภาษาไทยสวยงาม"
        result = word_tokenize(text, model="sefr-tnhc")
        self.assertEqual("".join(result), text)


class TestWordTokenizeSefrBest(unittest.TestCase):
    """Tests for the sefr-best model."""

    def test_basic_tokenization(self):
        result = word_tokenize("ทดสอบการตัดคำ", model="sefr-best")
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)
        self.assertEqual("".join(result), "ทดสอบการตัดคำ")

    def test_empty_string(self):
        result = word_tokenize("", model="sefr-best")
        self.assertEqual(result, [])

    def test_output_joins_to_input(self):
        text = "ภาษาไทยสวยงาม"
        result = word_tokenize(text, model="sefr-best")
        self.assertEqual("".join(result), text)


class TestWordTokenizeErrorHandling(unittest.TestCase):
    """Tests for error handling in word_tokenize."""

    def test_unsupported_model_raises(self):
        with self.assertRaises(NotImplementedError):
            word_tokenize("ทดสอบ", model="unknown-model")


if __name__ == "__main__":
    unittest.main()

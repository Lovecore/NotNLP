import unittest
from unittest.mock import patch, Mock
import file_nlp
import re

class TestYourMight(unittest.TestCase):

    @patch('file_nlp.BertTokenizer.from_pretrained')
    @patch('file_nlp.BertForSequenceClassification.from_pretrained')
    @patch('file_nlp.AutoTokenizer.from_pretrained')
    @patch('file_nlp.AutoModelForSequenceClassification.from_pretrained')

    # Probably don't need this test
    def test_classify_emotion(self, MockedAutoModel, MockedAutoTokenizer, MockedBertForSequenceClassification, MockedBertTokenizer):
        mock_model = Mock()
        mock_model.return_value.logits = 'fake_logits'
        MockedAutoModel.return_value = mock_model
        
        result, _ = file_nlp.classify_emotion("I am happy")
        self.assertIn(result, file_nlp.emotion_map.values())  # Checking if the result is within known emotion labels

    @patch('file_nlp.BertTokenizer.from_pretrained')
    @patch('file_nlp.BertForSequenceClassification.from_pretrained')

    # Probably don't really need this either
    def test_classify_sentiment(self, MockedBertForSequenceClassification, MockedBertTokenizer):
        mock_model = Mock()
        mock_model.return_value.logits = 'fake_logits'
        MockedBertForSequenceClassification.return_value = mock_model
        
        result = file_nlp.classify_sentiment("I am very happy")
        self.assertIn(result, file_nlp.sentiment_map.values())  # Checking if the result is within known sentiment labels
    
    def strip_ansi_codes(naked): 
        return re.sub(r'\x1B\[[0-?]*[ -/]*[@-~]', '', naked) # God damn colors

    def test_read_transcript_from_file(self):
        self.assertEqual(file_nlp.read_transcript_from_file('./tests/test_file.txt'), "test content")

    def test_load_trigger_words(self):
        self.assertEqual(file_nlp.load_trigger_words('./tests/test_triggers.txt'), ['word1', 'word2', 'word3', 'word4', 'word5', 'word6'])

    def test_format_emotion_probability(self):
        emotion_probabilities = {"anger": 0.1, "joy": 0.5, "surprise": 0.4}
        result = file_nlp.format_emotion_probability(emotion_probabilities, show_percent=True)
        stripped_result = strip_ansi_codes(result)
        self.assertIn("anger: 10.00%", stripped_result)
        self.assertIn("joy: 50.00%", stripped_result)
        self.assertIn("surprise: 40.00%", stripped_result)

if __name__ == '__main__':
    unittest.main()

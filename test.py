import json
import unittest
from mock import Mock, patch
import yaml
import os

from image_tagging import ImageTagger


class ImageTaggerTests(unittest.TestCase):

    ############################
    #### setup and teardown ####
    ############################
    helper = None

    # executed prior to each test
    def setUp(self):
        self.tagger = self.create_tagger()

    # executed after each test
    def tearDown(self):
        self.helper = None

    ########################
    #### helper methods ####
    ########################
    def create_tagger(self):
        tagger = ImageTagger()
        return tagger

    def configure_tagger(self, config_file, tagger):
        #Create the required params
        tagger.configure_tagger(config_file)

    ###############
    #### tests ####
    ###############
    def test_malformed_yaml_misconfigures_tagger(self):
        print 'Checking tagger is not configured...'
        # Empty config data

        d = {
            'visual-recognition': {
                'api-key': ''
            },
            'clarifai': {
                'client-id': '',
                'client-secret': ''
            }
        }
        with open('dummy_config.yml', 'w') as yaml_file:
            yaml_file.write(yaml.dump(d, default_flow_style=False))
        self.configure_tagger('dummy_config.yml', self.tagger)
        self.assertFalse(self.tagger.configured)
        os.remove('dummy_config.yml')

    def test_valid_yaml_configures_tagger(self):
        print 'Checking tagger is well configured...'
        self.configure_tagger('config.yml', self.tagger)
        self.assertTrue(self.tagger.configured)

    def test_configured_tagger_returns_data_from_visual_recognition_api(self):
        print 'Checking tagger queries visual recognition API'
        # Configure the mock to return a response with some dummy json response
        self.configure_tagger(config_file='config.yml', tagger=self.tagger)
        with patch('image_tagging.VisualRecognitionV3.classify') as mock_get:
            # Configure the mock to return a response with an OK status code.
            fd = open('fixtures/dummy_vr_result.json', 'r')
            dummy_response = json.load(fd)
            fd.close()
            mock_get.return_value = dummy_response
            response = self.tagger.process_images_visual_recognition('whatever', store_results=False)
            self.assertIsNotNone(response)
            self.assertTrue(response.empty)

    def test_configured_tagger_returns_data_from_clarifai(self):
        print 'Checking tagger queries clarifai API'
        # Configure the mock to return a response with some dummy json response
        self.configure_tagger(config_file='config.yml', tagger=self.tagger)
        with patch('image_tagging.ClarifaiApi.tag_images') as mock_get:
            # Configure the mock to return a response with an OK status code.
            fd = open('fixtures/dummy_clarifai_result.json', 'r')
            dummy_response = json.load(fd)
            fd.close()
            mock_get.return_value = dummy_response
            response = self.tagger.process_images_clarifai('whatever')
            import pdb
            pdb.set_trace()
            self.assertIsNotNone(response)
            self.assertTrue(response.empty)


if __name__ == "__main__":
    unittest.main()

import json
import unittest
import mock
from mock import Mock, patch
import yaml
import os
import pandas

from image_tagging import ImageTagger
from imagga import ImaggaHelper


class ImageTaggerTests(unittest.TestCase):

    ############################
    #### setup and teardown ####
    ############################
    helper = None
    google_vision_mocked_response_one = {u'responses': [{u'labelAnnotations': [{u'score': 0.93736339, u'mid': u'/m/01g317', u'description': u'person'}, {u'score': 0.83638608, u'mid': u'/m/07p82rh', u'description': u'human action'}, {u'score': 0.79835832, u'mid': u'/m/01lxd', u'description': u'coast'}, {u'score': 0.78364617, u'mid': u'/m/083mg', u'description': u'walking'}, {u'score': 0.77811265, u'mid': u'/m/0chml9', u'description': u'portrait photography'}]}, {u'labelAnnotations': [{u'score': 0.94670212, u'mid': u'/m/06_g7', u'description': u'surfing'}, {u'score': 0.88014328, u'mid': u'/m/034srq', u'description': u'wind wave'}, {u'score': 0.87241012, u'mid': u'/m/05kq4', u'description': u'ocean'}, {u'score': 0.8544699, u'mid': u'/m/06npx', u'description': u'sea'}, {u'score': 0.84946781, u'mid': u'/m/06ntj', u'description': u'sports'}]}, {u'labelAnnotations': [{u'score': 0.94382793, u'mid': u'/m/01g5v', u'description': u'blue'}, {u'score': 0.91175717, u'mid': u'/m/01sfb5', u'description': u'underwater'}, {u'score': 0.90479618, u'mid': u'/m/09kjpq', u'description': u'underwater diving'}, {u'score': 0.77733487, u'mid': u'/m/01p7j8', u'description': u'freediving'}, {u'score': 0.76457131, u'mid': u'/m/06npx', u'description': u'sea'}]}, {u'labelAnnotations': [{u'score': 0.95453954, u'mid': u'/m/02jwqh', u'description': u'vacation'}, {u'score': 0.94627291, u'mid': u'/m/06npx', u'description': u'sea'}, {u'score': 0.87351912, u'mid': u'/m/0b3yr', u'description': u'beach'}, {u'score': 0.85718411, u'mid': u'/m/05kq4', u'description': u'ocean'}, {u'score': 0.77249491, u'mid': u'/m/01r3xn', u'description': u'shoal'}]}, {u'labelAnnotations': [{u'score': 0.92269, u'mid': u'/m/0838f', u'description': u'water'}, {u'score': 0.91435242, u'mid': u'/m/01g5v', u'description': u'blue'}, {u'score': 0.85715073, u'mid': u'/m/06npx', u'description': u'sea'}, {u'score': 0.81179553, u'mid': u'/m/06ntj', u'description': u'sports'}, {u'score': 0.7820906, u'mid': u'/m/034srq', u'description': u'wind wave'}]}, {u'labelAnnotations': [{u'score': 0.67027062, u'mid': u'/m/04p25', u'description': u'loch'}, {u'score': 0.62062657, u'mid': u'/m/015kr', u'description': u'bridge'}, {u'score': 0.60307932, u'mid': u'/m/01jnfv', u'description': u'viaduct'}, {u'score': 0.5519985, u'mid': u'/m/07sn5gn', u'description': u'geological phenomenon'}, {u'score': 0.51492816, u'mid': u'/m/01lxd', u'description': u'coast'}]}, {u'labelAnnotations': [{u'score': 0.96301907, u'mid': u'/m/06npx', u'description': u'sea'}, {u'score': 0.93362737, u'mid': u'/m/05kq4', u'description': u'ocean'}, {u'score': 0.91102326, u'mid': u'/m/0d1n2', u'description': u'horizon'}, {u'score': 0.87906051, u'mid': u'/m/02fm9k', u'description': u'shore'}, {u'score': 0.84911305, u'mid': u'/m/02jwqh', u'description': u'vacation'}]}, {u'labelAnnotations': [{u'score': 0.98275393, u'mid': u'/m/036qh8', u'description': u'produce'}, {u'score': 0.95272052, u'mid': u'/m/02wbm', u'description': u'food'}, {u'score': 0.94636214, u'mid': u'/m/05s2s', u'description': u'plant'}, {u'score': 0.92663383, u'mid': u'/m/0f4s2w', u'description': u'vegetable'}, {u'score': 0.8451004, u'mid': u'/m/07j7r', u'description': u'tree'}]}, {u'labelAnnotations': [{u'score': 0.93530452, u'mid': u'/m/06_g7', u'description': u'surfing'}, {u'score': 0.89894521, u'mid': u'/m/0jp31', u'description': u'sail'}, {u'score': 0.88174415, u'mid': u'/m/07yv9', u'description': u'vehicle'}, {u'score': 0.87520057, u'mid': u'/m/01gg0y', u'description': u'windsurfing'}, {u'score': 0.84502131, u'mid': u'/m/03m4wn', u'description': u'boating'}]}, {u'labelAnnotations': [{u'score': 0.9614833, u'mid': u'/m/06npx', u'description': u'sea'}, {u'score': 0.92400092, u'mid': u'/m/05kq4', u'description': u'ocean'}, {u'score': 0.89314407, u'mid': u'/m/0d1n2', u'description': u'horizon'}, {u'score': 0.85161245, u'mid': u'/m/0b3yr', u'description': u'beach'}, {u'score': 0.84888953, u'mid': u'/m/01b2q6', u'description': u'sunrise'}]}]}
    google_vision_mocked_response_two = {u'responses': [{u'labelAnnotations': [{u'score': 0.97118294, u'mid': u'/m/06npx', u'description': u'sea'}, {u'score': 0.95370823, u'mid': u'/m/02fm9k', u'description': u'shore'}, {u'score': 0.95085073, u'mid': u'/m/05kq4', u'description': u'ocean'}, {u'score': 0.93212754, u'mid': u'/m/0b3yr', u'description': u'beach'}, {u'score': 0.93117511, u'mid': u'/m/01g5v', u'description': u'blue'}]}, {u'labelAnnotations': [{u'score': 0.949395, u'mid': u'/m/06npx', u'description': u'sea'}, {u'score': 0.94850063, u'mid': u'/m/02fm9k', u'description': u'shore'}, {u'score': 0.91411316, u'mid': u'/m/01g5v', u'description': u'blue'}, {u'score': 0.86089593, u'mid': u'/m/05kq4', u'description': u'ocean'}, {u'score': 0.83887005, u'mid': u'/m/01cbzq', u'description': u'rock'}]}, {u'labelAnnotations': [{u'score': 0.94408381, u'mid': u'/m/03q69', u'description': u'hair'}, {u'score': 0.93127322, u'mid': u'/m/01g317', u'description': u'person'}, {u'score': 0.92030478, u'mid': u'/m/0ds5b', u'description': u'facial hair'}, {u'score': 0.91022062, u'mid': u'/m/05qdh', u'description': u'painting'}, {u'score': 0.90178573, u'mid': u'/m/068jd', u'description': u'photograph'}]}, {u'labelAnnotations': [{u'score': 0.96552587, u'mid': u'/m/01g317', u'description': u'person'}, {u'score': 0.95651037, u'mid': u'/m/09x0r', u'description': u'speech'}, {u'score': 0.93644518, u'mid': u'/m/068k4', u'description': u'public speaking'}, {u'score': 0.80575955, u'mid': u'/m/063km', u'description': u'profession'}]}, {u'labelAnnotations': [{u'score': 0.9128592, u'mid': u'/m/07yv9', u'description': u'vehicle'}, {u'score': 0.85511458, u'mid': u'/m/01d74z', u'description': u'night'}, {u'score': 0.78026539, u'mid': u'/m/06gfj', u'description': u'road'}, {u'score': 0.73582464, u'mid': u'/m/0199g', u'description': u'bicycle'}, {u'score': 0.67002743, u'mid': u'/m/0ltv', u'description': u'auto racing'}]}, {u'labelAnnotations': [{u'score': 0.91870904, u'mid': u'/m/06npx', u'description': u'sea'}, {u'score': 0.88918173, u'mid': u'/m/01b2w5', u'description': u'sunset'}, {u'score': 0.84875685, u'mid': u'/m/01b2q6', u'description': u'sunrise'}, {u'score': 0.70815271, u'mid': u'/m/04k84', u'description': u'light'}, {u'score': 0.66666669, u'mid': u'/m/0gjyk', u'description': u'phenomenon'}]}, {u'labelAnnotations': [{u'score': 0.99229646, u'mid': u'/m/01yrx', u'description': u'cat'}, {u'score': 0.9876312, u'mid': u'/m/068hy', u'description': u'pet'}, {u'score': 0.97826087, u'mid': u'/m/01l7qd', u'description': u'whiskers'}, {u'score': 0.96503747, u'mid': u'/m/04rky', u'description': u'mammal'}, {u'score': 0.95658171, u'mid': u'/m/0jbk', u'description': u'animal'}]}, {u'labelAnnotations': [{u'score': 0.96784061, u'mid': u'/m/06npx', u'description': u'sea'}, {u'score': 0.93780887, u'mid': u'/m/01g5v', u'description': u'blue'}, {u'score': 0.91774017, u'mid': u'/m/05kq4', u'description': u'ocean'}, {u'score': 0.91310745, u'mid': u'/m/0838f', u'description': u'water'}, {u'score': 0.84560788, u'mid': u'/m/01phq4', u'description': u'pier'}]}, {u'labelAnnotations': [{u'score': 0.9339503, u'mid': u'/m/07yv9', u'description': u'vehicle'}, {u'score': 0.92866683, u'mid': u'/m/0k4j', u'description': u'automobile'}, {u'score': 0.81276566, u'mid': u'/m/088fh', u'description': u'yellow'}, {u'score': 0.73710233, u'mid': u'/m/07r04', u'description': u'truck'}, {u'score': 0.71874958, u'mid': u'/m/01prls', u'description': u'land vehicle'}]}, {u'labelAnnotations': [{u'score': 0.95438844, u'mid': u'/m/02jwqh', u'description': u'vacation'}, {u'score': 0.95418781, u'mid': u'/m/02n6m5', u'description': u'sun tanning'}, {u'score': 0.86673164, u'mid': u'/m/01m0lg', u'description': u'human positions'}, {u'score': 0.81480879, u'mid': u'/m/06npx', u'description': u'sea'}, {u'score': 0.75189954, u'mid': u'/m/0b3yr', u'description': u'beach'}]}]}
    google_vision_mocked_response_three = {u'responses': [{u'labelAnnotations': [{u'score': 0.92671192, u'mid': u'/m/06npx', u'description': u'sea'}, {u'score': 0.9003455, u'mid': u'/m/02jwqh', u'description': u'vacation'}, {u'score': 0.85687375, u'mid': u'/m/020dwk', u'description': u'passenger ship'}, {u'score': 0.82860547, u'mid': u'/m/03218s', u'description': u'dock'}, {u'score': 0.82054853, u'mid': u'/m/01jm2n', u'description': u'walkway'}]}, {u'labelAnnotations': [{u'score': 0.96000737, u'mid': u'/m/06ntj', u'description': u'sports'}, {u'score': 0.93333334, u'mid': u'/m/061bh', u'description': u'parasailing'}, {u'score': 0.92481387, u'mid': u'/m/01b7b1', u'description': u'kitesurfing'}, {u'score': 0.86427778, u'mid': u'/m/06npx', u'description': u'sea'}, {u'score': 0.78770655, u'mid': u'/m/05kq4', u'description': u'ocean'}]}, {u'labelAnnotations': [{u'score': 0.95097393, u'mid': u'/m/02jwqh', u'description': u'vacation'}, {u'score': 0.68116069, u'mid': u'/m/06npx', u'description': u'sea'}]}, {u'labelAnnotations': [{u'score': 0.91666669, u'mid': u'/m/0gd2v', u'description': u'marine mammal'}, {u'score': 0.89704657, u'mid': u'/m/01sfb5', u'description': u'underwater'}, {u'score': 0.8937847, u'mid': u'/m/050vn', u'description': u'marine biology'}, {u'score': 0.8869226, u'mid': u'/m/0jbk', u'description': u'animal'}, {u'score': 0.86636585, u'mid': u'/m/01540', u'description': u'biology'}]}, {u'labelAnnotations': [{u'score': 0.95660532, u'mid': u'/m/06npx', u'description': u'sea'}, {u'score': 0.91270089, u'mid': u'/m/0838f', u'description': u'water'}, {u'score': 0.89770734, u'mid': u'/m/05kq4', u'description': u'ocean'}, {u'score': 0.81240129, u'mid': u'/m/06ntj', u'description': u'sports'}, {u'score': 0.7637769, u'mid': u'/m/01gg0y', u'description': u'windsurfing'}]}]}
    google_mocked_list = [google_vision_mocked_response_one,
                          google_vision_mocked_response_two,
                          google_vision_mocked_response_three]
    # Indicates if all tests must be executed
    RUN_ALL_TESTS = False

    # executed prior to each test
    def setUp(self):
        self.tagger = self.create_tagger()
        self.imagga_helper = self.create_imagga_helper()


    # executed after each test
    def tearDown(self):
        self.helper = None

    ########################
    #### helper methods ####
    ########################
    def create_tagger(self):
        tagger = ImageTagger()
        return tagger

    def create_imagga_helper(self):
        imagga = ImaggaHelper()
        return imagga

    def configure_tagger(self, config_file, tagger):
        #Create the required params
        tagger.configure_tagger(config_file)

    def configure_imagga_helper(self, config_file, imagga_helper):
        #Create the required params
        imagga_helper.configure_imagga_helper(config_file)

    def mocked_requests(*args, **kwargs):
        """
        Used to mock the post response from uploading images to Imagga
        :param args: The URL for the request
        :param kwargs:  Any extra params, typically the image to be uploaded
        :return: A mocked dictionary
        """
        class MockResponse:
            def __init__(self, json_data, status_code):
                self.json_data = json_data
                self.status_code = status_code

            def json(self):
                return self.json_data

        if args[0] == 'https://api.imagga.com/v1/content':
            # Get the filename tried to uploaded
            response = {"status": "success",
                        "uploaded": [{"id": "4598e39043b2f7bbef85a44422fbd824",
                                       "filename": "sea-man-person-surfer.jpg"}]}
            return MockResponse(response, 200)
        elif args[0] == 'https://api.imagga.com/v1/tagging':
            # Get the filename tried to uploaded
            fd = open('fixtures/dummy_imagga_result.json', 'r')
            dummy_response = json.load(fd)
            fd.close()
            return MockResponse(dummy_response, 200)

        return MockResponse({}, 404)

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
            },
            'imagga': {
                'api-key': '',
                'api-secret': ''
            },
            'google-vision': {
                'api-key': ''
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
        # Patch the actual code that clarifai client uses
        with patch('clarifai.client.mime_util.post_multipart_request') as mock_get:
            # Configure the mock to return a response with an OK status code.
            fd = open('fixtures/dummy_clarifai_result.json', 'r')
            # Read as string
            dummy_response = fd.read()
            fd.close()
            mock_get.return_value = dummy_response
            response = self.tagger.process_images_clarifai(folder_name='sample_images')
            self.assertIsNotNone(response)
            # Check returned DataFrame has 25 rows, as expected
            self.assertEqual(25, len(response.index))

    def test_configured_imagga_wrapper_has_credentials(self):
        print 'Checking imagga helper has credentials upon configuration'
        # Configure the mock to return a response with some dummy json response
        self.configure_imagga_helper(config_file='config.yml', imagga_helper=self.imagga_helper)
        self.assertIsNotNone(self.imagga_helper.IMAGGA_API_KEY)
        self.assertIsNotNone(self.imagga_helper.IMAGGA_API_SECRET)

    @mock.patch('imagga.requests.post', side_effect=mocked_requests)
    def test_configured_imagga_wrapper_can_upload_image(self, mock_post):
        print 'Checking imagga helper can upload imaga to Imagga'
        # Configure the mock to return a response with some dummy json response
        self.configure_imagga_helper(config_file='config.yml', imagga_helper=self.imagga_helper)
        response = self.imagga_helper.upload_image(image_path='sample_images/sea-man-person-surfer.jpg')
        self.assertIsNotNone(response)
        self.assertEqual(response, u'4598e39043b2f7bbef85a44422fbd824')

    @mock.patch('imagga.requests.get', side_effect=mocked_requests)
    def test_configured_imagga_wrapper_can_tag_image(self, mock_post):
        print 'Checking imagga helper can tag Image'
        # Configure the mock to return a response with some dummy json response
        self.configure_imagga_helper(config_file='config.yml', imagga_helper=self.imagga_helper)
        response = self.imagga_helper.tag_image(image='4598e39043b2f7bbef85a44422fbd824')
        self.assertIsNotNone(response)
        self.assertIn('results', response)
        self.assertEqual('03204bdaaa3301fb8ce8a4c1e7a1ff17', response['results'][0]['image'])

    def test_configured_imagga_wrapper_can_tag_folder(self):
        print 'Checking imagga helper can tag a folder with images'
        # Configure the mock to return a response with some dummy json response
        self.configure_imagga_helper(config_file='config.yml', imagga_helper=self.imagga_helper)
        with patch('imagga.ImaggaHelper.tag_folder') as mock_get:
            # Configure the mock to return a response with an OK status code.
            fd = open('fixtures/dummy_imagga_total_results.json', 'r')
            # Read as string
            dummy_response = json.load(fd)
            fd.close()
            mock_get.return_value = dummy_response
            response = self.imagga_helper.tag_folder(folder_name='whatever')
            self.assertIsNotNone(response)
            self.assertEqual(25, len(response.keys()))

    def test_configured_imagga_wrapper_can_process_images(self):
        print 'Checking imagga helper can process images'
        # Configure the mock to return a response with some dummy json response
        self.configure_imagga_helper(config_file='config.yml', imagga_helper=self.imagga_helper)
        with patch('imagga.ImaggaHelper.tag_folder') as mock_get:
            # Configure the mock to return a response with an OK status code.
            fd = open('fixtures/dummy_imagga_total_results.json', 'r')
            # Read as string
            dummy_response = fd.read()
            fd.close()
            mock_get.return_value = dummy_response
            response = self.imagga_helper.tag_folder(folder_name='whatever')
            self.assertIsNotNone(response)
            # Call process images
            processed_images = self.imagga_helper.process_images(folder_name='whatever')
            # Check returned DataFrame has 16 rows, as expected
            self.assertEqual(25, len(processed_images.index))

    # @mock.patch('googleapiclient.http.HttpRequest.execute', side_effect=google_mocked_list)
    @unittest.skipUnless(RUN_ALL_TESTS, "All tests flag is not True")
    def test_configured_google_service_can_process_images(self, mock_post):
        print 'Checking Google Vision can label a folder with images'
        # Configure the mock to return a response with some dummy json response
        self.configure_tagger(config_file='config.yml', tagger=self.tagger)
        response = self.tagger.process_images_google_vision(folder_name='sample_images')
        self.assertIsNotNone(response)
        self.assertEqual(25, len(response.index))

    def test_all_apis_return_data(self):
        print 'Checking DataFrames return data'
        data_frame_imagga = None
        data_frame_google = None
        data_frame_clarifai = None
        data_frame_visual_recognition = None

        # Get data from Imagga
        with patch('imagga.ImaggaHelper.tag_folder') as mock_get:
            self.configure_imagga_helper(config_file='config.yml', imagga_helper=self.imagga_helper)
            # Configure the mock to return a response with an OK status code.
            fd = open('fixtures/dummy_imagga_total_results.json', 'r')
            # Read as string
            dummy_response = json.load(fd)
            fd.close()
            imagga_json_str = json.dumps(dummy_response, ensure_ascii=False, indent=4).encode('utf-8')
            mock_get.return_value = imagga_json_str
            # Call process images
            data_frame_imagga = self.imagga_helper.process_images(folder_name='whatever')
            # Check returned DataFrame has 16 rows, as expected
            self.assertEqual(25, len(data_frame_imagga.index))

        # Get data from Google
        self.configure_tagger(config_file='config.yml', tagger=self.tagger)
        with patch('googleapiclient.http.HttpRequest.execute', side_effect=ImageTaggerTests.google_mocked_list) as mock_get:
            data_frame_google = self.tagger.process_images_google_vision(folder_name='sample_images')
            self.assertIsNotNone(data_frame_google)
            self.assertEqual(25, len(data_frame_google.index))

        # Get data from clarifai
        with patch('clarifai.client.mime_util.post_multipart_request') as mock_get:
            # Configure the mock to return a response with an OK status code.
            fd = open('fixtures/dummy_clarifai_result.json', 'r')
            # Read as string
            dummy_response = fd.read()
            fd.close()
            mock_get.return_value = dummy_response
            data_frame_clarifai = self.tagger.process_images_clarifai(folder_name='sample_images')
            self.assertIsNotNone(data_frame_clarifai)
            # Check returned DataFrame has 25 rows, as expected
            self.assertEqual(25, len(data_frame_clarifai.index))

        # Get data from visual_recognition
        with patch('image_tagging.VisualRecognitionV3.classify') as mock_get:
            # Configure the mock to return a response with an OK status code.
            fd = open('fixtures/dummy_vr_result.json', 'r')
            dummy_response = json.load(fd)
            fd.close()
            mock_get.return_value = dummy_response
            data_frame_visual_recognition = self.tagger.process_images_visual_recognition('sample_images', store_results=False)
            self.assertIsNotNone(data_frame_visual_recognition)
            self.assertEqual(25, len(data_frame_visual_recognition.index))

        merged_df = pandas.concat([data_frame_imagga,
                                   data_frame_google,
                                   data_frame_clarifai,
                                   data_frame_visual_recognition],
                                  axis=1)
        self.assertEqual(4, len(merged_df.columns))
        self.assertEqual(25, len(merged_df.index))

if __name__ == "__main__":
    unittest.main()

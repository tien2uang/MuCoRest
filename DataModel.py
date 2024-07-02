import json
import numbers
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def is_response_content_similar(api_call_1, api_call_2) -> bool:
    from deepdiff import DeepDiff

    def deep_diff(dict1, dict2):
        """
      This function computes the deep difference between two dictionaries.

      Args:
        dict1: The first dictionary.
        dict2: The second dictionary.

      Returns:
        A dictionary containing the detailed differences between the two dictionaries.
      """

        diff = DeepDiff(dict1, dict2)
        if "dictionary_item_added" in diff or "dictionary_item_removed" in diff or "type_changes" in diff:
            return True
        else:
            return False

    from difflib import HtmlDiff

    def is_structure_similar(text1, text2):
        """
        Compares the structure of two HTML strings using difflib.HtmlDiff.

        Args:
            text1: First HTML string.
            text2: Second HTML string.

        Returns:
            A string containing the HTML difference between the two structures.
        """
        d = HtmlDiff()

        html_diff = d.html_diff(text1.splitlines(1), text2.splitlines(1))
        if "line_added" in html_diff or "line_removed" in html_diff:
            return False
        else:
            return True

    is_equal = True

    if api_call_1.response_type != api_call_2.response_type:
        is_equal = False
        return is_equal
    else:
        if (api_call_1.response_type == ReponseType.NO_CONTENT_RESPONSE_TYPE):
            response1 = api_call_1.response
            response2 = api_call_2.response
            if response1.status_code != response2.status_code:
                is_equal = False
                return is_equal
        elif api_call_1.response_type == ReponseType.HTML_RESPONSE_TYPE:
            return is_structure_similar(api_call_1.text, api_call_2.text)

        elif api_call_1.response_type == ReponseType.JSON_RESPONSE_TYPE:
            response1 = api_call_1.response
            response2 = api_call_2.response
            if response1.status_code != response2.status_code:
                is_equal = False
                return is_equal
            try:

                if (response1.json() is not None) and (response2.json() is not None):
                    if type(response1.json()) != type(response2.json()):
                        is_equal = False
                        return is_equal
                    else:

                        if isinstance(response1.json(), dict) or isinstance(response1.json(), list):

                            # if get_depth(response1.json()) != get_depth(response2.json()):
                            #    is_equal = False

                            if deep_diff(response1.json(), response2.json()):
                                is_equal = False
            except requests.exceptions.JSONDecodeError as e:
                print("Invalid JSON response:", e)
                with open('response_fail.txt', 'w') as f:
                    f.write(response1.text + "\n")
                    f.write(response2.text + "\n")
        elif api_call_1.response_type == ReponseType.OTHER_RESPONSE_TYPE:

            response1 = api_call_1.response
            response2 = api_call_2.response
            if response1.status_code != response2.status_code:
                is_equal = False
                return is_equal

            def count_words(text):

                # Split the string into words using the split() method
                words = text.split()

                # Count the number of words
                word_count = len(words)

                return word_count

            if isinstance(response1.text, str):
                return True if count_words(response1.text) == count_words(response2.text) else False


        else:
            return True
    return is_equal


class ReponseType:
    HTML_RESPONSE_TYPE = 1
    JSON_RESPONSE_TYPE = 2
    NO_CONTENT_RESPONSE_TYPE = 3
    OTHER_RESPONSE_TYPE = 4


class APICall:

    def __init__(self, operation, response):
        self.operation = operation
        self.response = response
        if (len(response.text) == 0):
            self.response_type = ReponseType.NO_CONTENT_RESPONSE_TYPE
        elif (response.text[0] == '<'):
            self.response_type = ReponseType.HTML_RESPONSE_TYPE
        elif (response.text[0] == '[' and response.text[-1] == ']') or (
                response.text[0] == '{' and response.text[-1] == '}'):
            self.response_type = ReponseType.JSON_RESPONSE_TYPE
        else:
            self.response_type = ReponseType.OTHER_RESPONSE_TYPE


class FIFOListWithMaxSize:
    def __init__(self, max_size):
        self.max_size = max_size
        self._list = []

    def append(self, item):
        if len(self._list) == self.max_size:
            self._list.pop(0)  # Remove the oldest item to make space
        self._list.append(item)

    def pop(self):
        if not self._list:
            raise IndexError("pop from empty FIFOListWithMaxSize")
        return self._list.pop(0)

    def __len__(self):
        return len(self._list)

    def __str__(self):
        return str(self._list)  # For easy visualization


class APICallList(FIFOListWithMaxSize):
    def __init__(self, max_size=5):
        super().__init__(max_size)

    def count_similar_api_call(self, target_api_call) -> int:
        if len(self._list) == 0:
            return 0
        else:
            count = 0
            for api_call in self._list:
                if is_response_content_similar(api_call, target_api_call):
                    count += 1

        return count

    def append(self, api_call: APICall):
        if api_call.response_type == ReponseType.JSON_RESPONSE_TYPE:
            try:
                data = api_call.response.json()
                super().append(api_call)
                print("Success add to ", api_call.operation['operation_id'], api_call.response.status_code)
            except requests.exceptions.JSONDecodeError as e:
                print("Invalid JSON response:", e)
                print("Cannot append to API List ", api_call.operation['operation_id'], api_call.response.status_code)
                with open('response_fail.txt', 'a') as f:
                    f.write(api_call.response.text + "\n")
                print("-----------------------------------")

    def get_all(self):
        return self._list


class LineCoverageList(FIFOListWithMaxSize):
    def __init__(self, max_size=5):
        super().__init__(max_size)

    def calculate_relative_coverage(self, coverage) -> float:

        if self.__len__() == 1:
            return (coverage - self._list[0]) / 0.05
        else:

            average_jump_distance = (self._list[-1] - self._list[0]) / (self.__len__() - 1)
            if average_jump_distance == 0:
                return (coverage - self._list[-1]) / 0.05
            else:
                return (coverage - self._list[-1]) / average_jump_distance


class StackTraceList(FIFOListWithMaxSize):
    def __init__(self, max_size=5):
        super().__init__(max_size)

    def count_exact_stack_trace_matches(self, target_stack_trace) -> int:
        similar_st_quantity = 0
        if len(self._list) == 0:
            return similar_st_quantity
        else:
            target_stack_trace_length = len(target_stack_trace)
            for stack_trace in self._list:
                if target_stack_trace == stack_trace:
                    similar_st_quantity += 1

        return similar_st_quantity

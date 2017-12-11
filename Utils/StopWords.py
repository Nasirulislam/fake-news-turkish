"""
Utility class to filter our stop words
"""


class StopWords:
    def __init__(self):
        # read the stop words from the files
        with open("stop_words.txt", "r") as f:
            all_words = f.read().decode("utf-8").splitlines()
            # remove comments that start with #
            all_words = filter(lambda w: not w.startswith("#"), all_words)
            self.stop_words = all_words

    def remove_stop_words(self, input):
        """
        Remove the words that are stop words from the input list of words
        :param input: A list of words
        :return: new list of words that does not contain stop words
        """
        return [word for word in input if word not in self.stop_words]

from kotaemon.indices.qa.utils import find_start_end_phrase

DOC = "Sargon of Akkad founded the empire. It ruled Mesopotamia from Akkad."


def test_find_start_end_phrase_accepts_none():
    assert find_start_end_phrase("Sargon of Akkad", None, DOC) == ((0, 15), 15)
    assert find_start_end_phrase(None, None, DOC) == (None, 0)

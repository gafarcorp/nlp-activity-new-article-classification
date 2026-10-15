from backend.app.preprocessing import basic_clean, clean_text, fix_entities


def test_fix_entities_handles_both_entity_forms():
    assert fix_entities("it#39;s") == "it's"                       # '&' stripped in the dataset
    assert fix_entities("Tom &amp; Jerry") == "Tom & Jerry"        # normal entity
    assert fix_entities('He said quot;no quot;') == 'He said "no "'


def test_entity_leftovers_do_not_become_tokens():
    cleaned = clean_text("Stocks rise it#39;s up &lt;b&gt;fast&lt;/b&gt; quot;wow quot;")
    for junk in ("39", "lt", "gt", "quot"):
        assert junk not in cleaned.split()


def test_basic_clean_keeps_case_and_punctuation_for_bert():
    out = basic_clean("Apple <b>Inc.</b> reports\\profit, shares rise!")
    assert "Apple" in out and "," in out and "!" in out
    assert "<b>" not in out and "\\" not in out


def test_words_ending_in_lt_are_not_mangled():
    assert "result;" in fix_entities("result; default;")


def test_clean_text_removes_stopwords_and_lemmatizes():
    assert clean_text("The markets are rising") == "market rising"

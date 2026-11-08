from codesearch.tokens import looks_like_symbol, split_identifier, stem, terms


def test_split_snake_camel_pascal_and_acronyms():
    assert split_identifier("parse_http_response") == ["parse", "http", "response"]
    assert split_identifier("parseHTTPResponse") == ["parse", "http", "response"]
    assert split_identifier("HTTPServer") == ["http", "server"]
    assert split_identifier("__init__") == ["init"]
    assert split_identifier("utf8Decode") == ["utf", "8", "decode"]


def test_stem_equates_inflections():
    assert len({stem(w) for w in ["parse", "parsing", "parsed", "parses"]}) == 1
    assert len({stem(w) for w in ["cache", "caches", "cached", "caching"]}) == 1
    assert stem("retries") == stem("retry") == "retry"
    assert stem("class") == "class"  # -ss stays
    assert stem("is") == "is" and stem("bus") == "bus"


def test_stem_never_empties_short_words():
    for w in ["a", "as", "ing", "ed", "es", "s", "ies"]:
        assert stem(w)


def test_terms_drops_stopwords_digits_and_single_letters():
    assert terms("How do I parse the HTTP headers?") == [stem("parse"), "http", "header"]
    assert terms("x = 42") == []
    assert terms("retry_with_backoff") == ["retry", "backoff"]


def test_terms_is_identifier_and_prose_symmetric():
    assert terms("parseHTTPHeaders") == terms("parse http headers")


def test_looks_like_symbol():
    assert looks_like_symbol("parse_http_headers")
    assert looks_like_symbol("parseHTTPHeaders")
    assert looks_like_symbol("`slugify`")
    assert looks_like_symbol("slugify()")
    assert looks_like_symbol("Foo.bar_baz")
    assert not looks_like_symbol("slugify")
    assert not looks_like_symbol("hello,")
    assert not looks_like_symbol("a-b")

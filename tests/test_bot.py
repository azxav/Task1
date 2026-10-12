from faq.bot import response_messages, split_message


def test_long_answers_are_complete_and_fit_telegram_limit():
    text = "🙂" * 3000 + "\n" + "a" * 5000
    chunks = split_message(text)
    assert "".join(chunks) == text
    assert all(len(chunk.encode("utf-16-le")) // 2 <= 4096 for chunk in chunks)


def test_bot_displays_stored_answers_and_no_match_messages():
    assert response_messages({"status": "matched", "answer": "Stored verbatim."}) == ["Stored verbatim."]
    assert response_messages({"status": "no_match", "message": "Aniqroq yozing."}) == ["Aniqroq yozing."]

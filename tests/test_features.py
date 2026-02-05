"""Tests for WhisperFlow features - unit tests for individual components."""

import json
import os
import tempfile
import time

import pytest


# ============================================================================
# Personal Dictionary
# ============================================================================

class TestPersonalDictionary:
    def setup_method(self):
        self.tmpfile = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.tmpfile.write(b"{}")
        self.tmpfile.close()

    def teardown_method(self):
        os.unlink(self.tmpfile.name)

    def test_add_and_get_words(self):
        from whisperflow.features.dictionary import PersonalDictionary
        d = PersonalDictionary(self.tmpfile.name)
        d.add_word("TensorFlow", "technical")
        d.add_word("Kubernetes", "technical")
        words = d.get_words()
        assert "TensorFlow" in words
        assert words["TensorFlow"] == "technical"
        assert len(words) == 2

    def test_remove_word(self):
        from whisperflow.features.dictionary import PersonalDictionary
        d = PersonalDictionary(self.tmpfile.name)
        d.add_word("test", "general")
        d.remove_word("test")
        assert "test" not in d.get_words()

    def test_remove_nonexistent_word(self):
        from whisperflow.features.dictionary import PersonalDictionary
        d = PersonalDictionary(self.tmpfile.name)
        d.remove_word("nonexistent")  # should not raise

    def test_whisper_prompt(self):
        from whisperflow.features.dictionary import PersonalDictionary
        d = PersonalDictionary(self.tmpfile.name)
        d.add_word("WhisperFlow")
        d.add_word("PyTorch")
        prompt = d.get_whisper_prompt()
        assert "WhisperFlow" in prompt
        assert "PyTorch" in prompt

    def test_empty_prompt(self):
        from whisperflow.features.dictionary import PersonalDictionary
        d = PersonalDictionary(self.tmpfile.name)
        assert d.get_whisper_prompt() == ""

    def test_persistence_across_instances(self):
        from whisperflow.features.dictionary import PersonalDictionary
        d1 = PersonalDictionary(self.tmpfile.name)
        d1.add_word("persisted", "test")
        d2 = PersonalDictionary(self.tmpfile.name)
        assert "persisted" in d2.get_words()

    def test_check_and_learn_proper_noun(self):
        from whisperflow.features.dictionary import PersonalDictionary
        d = PersonalDictionary(self.tmpfile.name)
        learned = d.check_and_learn(
            "i talked to john about the project",
            "I talked to John about the project",
        )
        assert "John" in learned
        assert "John" in d.get_words()

    def test_check_and_learn_camelcase(self):
        from whisperflow.features.dictionary import PersonalDictionary
        d = PersonalDictionary(self.tmpfile.name)
        learned = d.check_and_learn(
            "use the get data function",
            "use the getData function",
        )
        assert "getData" in learned


# ============================================================================
# Snippets
# ============================================================================

class TestSnippetManager:
    def setup_method(self):
        self.tmpfile = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.tmpfile.write(b"{}")
        self.tmpfile.close()

    def teardown_method(self):
        os.unlink(self.tmpfile.name)

    def test_add_and_trigger(self):
        from whisperflow.features.snippets import SnippetManager
        s = SnippetManager(self.tmpfile.name)
        s.add_snippet("my calendar link", "https://calendly.com/user")
        result = s.check_for_snippet("my calendar link")
        assert result == "https://calendly.com/user"

    def test_trigger_with_prefix(self):
        from whisperflow.features.snippets import SnippetManager
        s = SnippetManager(self.tmpfile.name)
        s.add_snippet("my email", "user@example.com")
        assert s.check_for_snippet("insert my email") == "user@example.com"
        assert s.check_for_snippet("paste my email") == "user@example.com"
        assert s.check_for_snippet("add my email") == "user@example.com"
        assert s.check_for_snippet("type my email") == "user@example.com"

    def test_no_match(self):
        from whisperflow.features.snippets import SnippetManager
        s = SnippetManager(self.tmpfile.name)
        assert s.check_for_snippet("hello world") is None

    def test_remove_snippet(self):
        from whisperflow.features.snippets import SnippetManager
        s = SnippetManager(self.tmpfile.name)
        s.add_snippet("test", "value")
        s.remove_snippet("test")
        assert s.check_for_snippet("test") is None

    def test_case_insensitive_trigger(self):
        from whisperflow.features.snippets import SnippetManager
        s = SnippetManager(self.tmpfile.name)
        s.add_snippet("My Email", "user@example.com")
        assert s.check_for_snippet("my email") == "user@example.com"

    def test_fuzzy_match_in_sentence(self):
        from whisperflow.features.snippets import SnippetManager
        s = SnippetManager(self.tmpfile.name)
        s.add_snippet("disclaimer", "This is not legal advice.")
        assert s.check_for_snippet("add the disclaimer here") == "This is not legal advice."

    def test_persistence(self):
        from whisperflow.features.snippets import SnippetManager
        s1 = SnippetManager(self.tmpfile.name)
        s1.add_snippet("sig", "Best regards, User")
        s2 = SnippetManager(self.tmpfile.name)
        assert s2.check_for_snippet("sig") == "Best regards, User"

    def test_get_snippets(self):
        from whisperflow.features.snippets import SnippetManager
        s = SnippetManager(self.tmpfile.name)
        s.add_snippet("a", "content a")
        s.add_snippet("b", "content b")
        snippets = s.get_snippets()
        assert len(snippets) == 2
        assert "a" in snippets
        assert "b" in snippets


# ============================================================================
# History
# ============================================================================

class TestTranscriptionHistory:
    def setup_method(self):
        self.tmpfile = tempfile.NamedTemporaryFile(suffix=".db", delete=False)
        self.tmpfile.close()

    def teardown_method(self):
        os.unlink(self.tmpfile.name)

    def test_add_and_get(self):
        from whisperflow.features.history import TranscriptionHistory
        h = TranscriptionHistory(self.tmpfile.name)
        h.add("raw text", "processed text", app_context="slack", mode="dictation")
        entries = h.get_recent()
        assert len(entries) == 1
        assert entries[0].processed_text == "processed text"
        assert entries[0].app_context == "slack"
        assert entries[0].mode == "dictation"

    def test_search(self):
        from whisperflow.features.history import TranscriptionHistory
        h = TranscriptionHistory(self.tmpfile.name)
        h.add("hello", "Hello!", app_context="email")
        h.add("code fix", "Fixed the bug.", app_context="code")
        results = h.search("bug")
        assert len(results) == 1
        assert "bug" in results[0].processed_text.lower()

    def test_max_entries(self):
        from whisperflow.features.history import TranscriptionHistory
        h = TranscriptionHistory(self.tmpfile.name, max_entries=5)
        for i in range(10):
            h.add(f"raw {i}", f"processed {i}")
        assert h.count <= 5

    def test_get_entry_by_id(self):
        from whisperflow.features.history import TranscriptionHistory
        h = TranscriptionHistory(self.tmpfile.name)
        entry_id = h.add("raw", "processed")
        entry = h.get_entry(entry_id)
        assert entry is not None
        assert entry.raw_text == "raw"

    def test_delete_entry(self):
        from whisperflow.features.history import TranscriptionHistory
        h = TranscriptionHistory(self.tmpfile.name)
        entry_id = h.add("raw", "processed")
        h.delete_entry(entry_id)
        assert h.get_entry(entry_id) is None

    def test_clear(self):
        from whisperflow.features.history import TranscriptionHistory
        h = TranscriptionHistory(self.tmpfile.name)
        for i in range(5):
            h.add(f"raw {i}", f"processed {i}")
        h.clear()
        assert h.count == 0

    def test_ordering_newest_first(self):
        from whisperflow.features.history import TranscriptionHistory
        h = TranscriptionHistory(self.tmpfile.name)
        h.add("first", "first")
        time.sleep(0.01)
        h.add("second", "second")
        entries = h.get_recent()
        assert entries[0].raw_text == "second"
        assert entries[1].raw_text == "first"

    def test_search_no_results(self):
        from whisperflow.features.history import TranscriptionHistory
        h = TranscriptionHistory(self.tmpfile.name)
        h.add("hello", "Hello!")
        results = h.search("nonexistent")
        assert len(results) == 0

    def test_multiple_fields_stored(self):
        from whisperflow.features.history import TranscriptionHistory
        h = TranscriptionHistory(self.tmpfile.name)
        h.add("raw", "processed", app_context="code", mode="command", language="en", duration=2.5)
        entry = h.get_recent(1)[0]
        assert entry.app_context == "code"
        assert entry.mode == "command"
        assert entry.language == "en"
        assert entry.duration == 2.5


# ============================================================================
# Auto-Editor
# ============================================================================

class TestAutoEditor:
    def test_basic_cleanup_short_text(self):
        from whisperflow.processing.auto_edit import AutoEditor
        from whisperflow.llm.factory import NoopProvider

        editor = AutoEditor(NoopProvider())
        result = editor._basic_cleanup("um hello")
        assert "um" not in result.lower()
        assert "hello" in result.lower()

    def test_empty_input(self):
        from whisperflow.processing.auto_edit import AutoEditor
        from whisperflow.llm.factory import NoopProvider

        editor = AutoEditor(NoopProvider())
        assert editor.process("") == ""
        assert editor.process("   ") == ""

    def test_basic_cleanup_adds_period(self):
        from whisperflow.processing.auto_edit import AutoEditor
        from whisperflow.llm.factory import NoopProvider

        editor = AutoEditor(NoopProvider())
        result = editor._basic_cleanup("hello world")
        assert result.endswith(".")

    def test_basic_cleanup_preserves_existing_punctuation(self):
        from whisperflow.processing.auto_edit import AutoEditor
        from whisperflow.llm.factory import NoopProvider

        editor = AutoEditor(NoopProvider())
        result = editor._basic_cleanup("is this a question?")
        assert not result.endswith("?.")  # should not double-punctuate

    def test_basic_cleanup_capitalizes(self):
        from whisperflow.processing.auto_edit import AutoEditor
        from whisperflow.llm.factory import NoopProvider

        editor = AutoEditor(NoopProvider())
        result = editor._basic_cleanup("hello world")
        assert result[0].isupper()

    def test_basic_cleanup_removes_fillers(self):
        from whisperflow.processing.auto_edit import AutoEditor
        from whisperflow.llm.factory import NoopProvider

        editor = AutoEditor(NoopProvider())
        result = editor._basic_cleanup("um uh basically thinking you know about this")
        assert "um" not in result.lower().split()
        assert "uh" not in result.lower().split()

    def test_noop_provider_passthrough_for_long_text(self):
        from whisperflow.processing.auto_edit import AutoEditor
        from whisperflow.llm.factory import NoopProvider

        editor = AutoEditor(NoopProvider())
        text = "this is a longer sentence that should go through the LLM path"
        result = editor.process(text)
        assert result == text

    def test_process_short_text_uses_basic_cleanup(self):
        from whisperflow.processing.auto_edit import AutoEditor
        from whisperflow.llm.factory import NoopProvider

        editor = AutoEditor(NoopProvider())
        result = editor.process("hi")
        # Short text (<=3 words) goes through basic cleanup
        assert result[0].isupper()


# ============================================================================
# Command Mode
# ============================================================================

class TestCommandMode:
    def test_classify_question(self):
        from whisperflow.processing.command_mode import CommandMode
        from whisperflow.llm.factory import NoopProvider

        cmd = CommandMode(NoopProvider())
        assert cmd.classify_intent("what is the capital of France") == "question"
        assert cmd.classify_intent("how do I install Python") == "question"
        assert cmd.classify_intent("who is the president") == "question"
        assert cmd.classify_intent("explain how recursion works") == "question"
        assert cmd.classify_intent("tell me about quantum computing") == "question"

    def test_classify_command_keywords(self):
        from whisperflow.processing.command_mode import CommandMode
        from whisperflow.llm.factory import NoopProvider

        cmd = CommandMode(NoopProvider())
        assert cmd.classify_intent("translate this to Spanish") == "command"
        assert cmd.classify_intent("make this more formal") == "command"
        assert cmd.classify_intent("summarize this") == "command"
        assert cmd.classify_intent("turn this into bullet points") == "command"
        assert cmd.classify_intent("rewrite this paragraph") == "command"
        assert cmd.classify_intent("simplify this text") == "command"

    def test_execute_command_no_selected_text(self):
        from whisperflow.processing.command_mode import CommandMode
        from whisperflow.llm.factory import NoopProvider

        cmd = CommandMode(NoopProvider())
        result = cmd.execute_command("make this formal", "")
        assert result == ""

    def test_execute_command_with_text(self):
        from whisperflow.processing.command_mode import CommandMode
        from whisperflow.llm.factory import NoopProvider

        cmd = CommandMode(NoopProvider())
        result = cmd.execute_command("make this formal", "hey whats up")
        # NoopProvider returns user_message; command gets formatted as user_message
        assert result  # should return something non-empty

    def test_answer_question(self):
        from whisperflow.processing.command_mode import CommandMode
        from whisperflow.llm.factory import NoopProvider

        cmd = CommandMode(NoopProvider())
        result = cmd.answer_question("what is Python")
        assert "what is Python" in result


# ============================================================================
# Window Context Detection
# ============================================================================

class TestWindowContext:
    def test_detected_app_email_thunderbird(self):
        from whisperflow.input.context import WindowContext
        ctx = WindowContext("thunderbird", "Inbox - Thunderbird", "Thunderbird")
        assert ctx.detected_app == "email"

    def test_detected_app_email_evolution(self):
        from whisperflow.input.context import WindowContext
        ctx = WindowContext("evolution", "Inbox - Evolution", "Evolution")
        assert ctx.detected_app == "email"

    def test_detected_app_slack(self):
        from whisperflow.input.context import WindowContext
        ctx = WindowContext("slack", "general - Slack", "Slack")
        assert ctx.detected_app == "slack"

    def test_detected_app_discord(self):
        from whisperflow.input.context import WindowContext
        ctx = WindowContext("discord", "Server - Discord", "Discord")
        assert ctx.detected_app == "discord"

    def test_detected_app_telegram(self):
        from whisperflow.input.context import WindowContext
        ctx = WindowContext("telegram", "Chat - Telegram", "Telegram")
        assert ctx.detected_app == "telegram"

    def test_detected_app_code(self):
        from whisperflow.input.context import WindowContext
        ctx = WindowContext("code", "main.py - Visual Studio Code", "Code")
        assert ctx.detected_app == "code"

    def test_detected_app_cursor(self):
        from whisperflow.input.context import WindowContext
        ctx = WindowContext("cursor", "main.py - Cursor", "Cursor")
        assert ctx.detected_app == "code"

    def test_detected_app_terminal(self):
        from whisperflow.input.context import WindowContext
        ctx = WindowContext("alacritty", "~/ - Alacritty", "Alacritty")
        assert ctx.detected_app == "terminal"

    def test_detected_app_browser_gmail(self):
        from whisperflow.input.context import WindowContext
        ctx = WindowContext("firefox", "Inbox - Gmail - Firefox", "Firefox")
        assert ctx.detected_app == "email"

    def test_detected_app_browser_slack(self):
        from whisperflow.input.context import WindowContext
        ctx = WindowContext("chrome", "general - Slack - Google Chrome", "Chrome")
        assert ctx.detected_app == "slack"

    def test_detected_app_browser_chatgpt(self):
        from whisperflow.input.context import WindowContext
        ctx = WindowContext("firefox", "ChatGPT - Firefox", "Firefox")
        assert ctx.detected_app == "ai-chat"

    def test_detected_app_browser_claude(self):
        from whisperflow.input.context import WindowContext
        ctx = WindowContext("firefox", "Claude - Firefox", "Firefox")
        assert ctx.detected_app == "ai-chat"

    def test_detected_app_browser_generic(self):
        from whisperflow.input.context import WindowContext
        ctx = WindowContext("firefox", "Some Website - Firefox", "Firefox")
        assert ctx.detected_app == "browser"

    def test_detected_app_libreoffice(self):
        from whisperflow.input.context import WindowContext
        ctx = WindowContext("soffice", "Document.odt - LibreOffice Writer", "soffice")
        assert ctx.detected_app == "document"

    def test_detected_app_unknown(self):
        from whisperflow.input.context import WindowContext
        ctx = WindowContext("randomapp", "Random", "Random")
        assert ctx.detected_app == "other"

    def test_empty_context(self):
        from whisperflow.input.context import WindowContext
        ctx = WindowContext("", "", "")
        assert ctx.detected_app == "other"


# ============================================================================
# Style Learning
# ============================================================================

class TestStyleLearning:
    def setup_method(self):
        self.tmpfile = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        self.tmpfile.write(b"{}")
        self.tmpfile.close()

    def teardown_method(self):
        os.unlink(self.tmpfile.name)

    def test_record_correction(self):
        from whisperflow.features.style_learning import StyleLearner
        sl = StyleLearner(self.tmpfile.name)
        sl.record_correction("Hello World.", "hello world")
        assert len(sl._examples) == 1

    def test_skip_identical_correction(self):
        from whisperflow.features.style_learning import StyleLearner
        sl = StyleLearner(self.tmpfile.name)
        sl.record_correction("Hello", "Hello")
        assert len(sl._examples) == 0

    def test_max_examples_limit(self):
        from whisperflow.features.style_learning import StyleLearner
        sl = StyleLearner(self.tmpfile.name, max_examples=5)
        for i in range(10):
            sl.record_correction(f"original {i}", f"corrected {i}")
        assert len(sl._examples) == 5

    def test_style_prompt_empty_when_no_examples(self):
        from whisperflow.features.style_learning import StyleLearner
        sl = StyleLearner(self.tmpfile.name)
        assert sl.get_style_prompt_addition() == ""

    def test_style_prompt_with_examples(self):
        from whisperflow.features.style_learning import StyleLearner
        sl = StyleLearner(self.tmpfile.name)
        sl.record_correction("Hello World.", "hey world")
        prompt = sl.get_style_prompt_addition()
        assert "Before:" in prompt
        assert "After:" in prompt

    def test_persistence(self):
        from whisperflow.features.style_learning import StyleLearner
        sl1 = StyleLearner(self.tmpfile.name)
        sl1.record_correction("original", "corrected")
        sl2 = StyleLearner(self.tmpfile.name)
        assert len(sl2._examples) == 1

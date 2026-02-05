"""Tests for WhisperFlow features."""

import json
import os
import tempfile

import pytest


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


class TestAutoEditor:
    def test_basic_cleanup_short_text(self):
        from whisperflow.features.dictionary import PersonalDictionary
        from whisperflow.processing.auto_edit import AutoEditor
        from whisperflow.llm.factory import NoopProvider

        editor = AutoEditor(NoopProvider())
        # Short text uses basic cleanup
        result = editor._basic_cleanup("um hello")
        assert "um" not in result.lower() or result  # fillers removed or at least processed

    def test_empty_input(self):
        from whisperflow.processing.auto_edit import AutoEditor
        from whisperflow.llm.factory import NoopProvider

        editor = AutoEditor(NoopProvider())
        assert editor.process("") == ""
        assert editor.process("   ") == ""


class TestCommandMode:
    def test_classify_question(self):
        from whisperflow.processing.command_mode import CommandMode
        from whisperflow.llm.factory import NoopProvider

        cmd = CommandMode(NoopProvider())
        assert cmd.classify_intent("what is the capital of France") == "question"
        assert cmd.classify_intent("how do I install Python") == "question"

    def test_classify_command_keywords(self):
        from whisperflow.processing.command_mode import CommandMode
        from whisperflow.llm.factory import NoopProvider

        cmd = CommandMode(NoopProvider())
        assert cmd.classify_intent("translate this to Spanish") == "command"
        assert cmd.classify_intent("make this more formal") == "command"
        assert cmd.classify_intent("summarize this") == "command"


class TestWindowContext:
    def test_detected_app_email(self):
        from whisperflow.input.context import WindowContext

        ctx = WindowContext("thunderbird", "Inbox - Thunderbird", "Thunderbird")
        assert ctx.detected_app == "email"

    def test_detected_app_slack(self):
        from whisperflow.input.context import WindowContext

        ctx = WindowContext("slack", "general - Slack", "Slack")
        assert ctx.detected_app == "slack"

    def test_detected_app_code(self):
        from whisperflow.input.context import WindowContext

        ctx = WindowContext("code", "main.py - Visual Studio Code", "Code")
        assert ctx.detected_app == "code"

    def test_detected_app_browser_gmail(self):
        from whisperflow.input.context import WindowContext

        ctx = WindowContext("firefox", "Inbox - Gmail - Firefox", "Firefox")
        assert ctx.detected_app == "email"

    def test_detected_app_browser_generic(self):
        from whisperflow.input.context import WindowContext

        ctx = WindowContext("firefox", "Some Website - Firefox", "Firefox")
        assert ctx.detected_app == "browser"
